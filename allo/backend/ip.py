# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import re
import importlib
import subprocess
import time
import tempfile

# Template argument list allowing one level of nesting, so that both `int8_t`
# and `ap_int<8>` work as the argument of an outer template.
_TEMPLATE_ARGS = r"[^<>]*(?:<[^<>]*>[^<>]*)*"

# Regex token that matches plain C types (int8_t, float), HLS-style template
# types (ap_int<8>, ap_uint<16>) and namespace-qualified templates as emitted
# by EmitVivadoHLS (`hls::stream< int8_t >`, `hls::stream< ap_int<8> >`).
# The template alternative must come first: otherwise the bare `\w+` branch
# would match only the head of `hls::stream<...>`.
_TYPE_TOKEN = rf"(?:(?:\w+::)*\w+\s*<{_TEMPLATE_ARGS}>|\w+)"

# An `hls::stream<T>`, which HLS code always passes by reference.
_STREAM_TOKEN = rf"(?:\w+::)*stream\s*<{_TEMPLATE_ARGS}>"

# Directory holding the CPU-simulation shim headers (`hls_stream.h`,
# `allo_fifo.h`). It is put FIRST on the include path of the simulator wrapper
# so that the IP's `#include <hls_stream.h>` resolves to Allo's shim instead of
# Vitis's header. See `docs/IP_STREAM_SIM_SHIM.md`.
IP_SIM_INCLUDE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ip_sim")

# The only symbol a generated wrapper .so may export. Everything else -- above
# all the IP's own top function, whose name is whatever the user wrote and is
# therefore the same in every .so built from it -- is compiled with
# `-fvisibility=hidden` (see `compile_shared_lib`).
#
# Why this matters: two IPModules built from *different* sources may share a top
# name (e.g. two tests that each generate their own `vadd_stream`). Both .so
# files land in one process, and the dynamic linker resolves a global symbol to
# the definition it saw FIRST -- so the second wrapper would silently call the
# first IP's body. Hiding it makes each wrapper's call bind inside its own .so.
_EXPORT_ATTR = '__attribute__((visibility("default")))'


def split_template_args(inner: str):
    """Split a template argument list on its top-level commas.

    ``"int32_t, 4"`` -> ``["int32_t", "4"]``, while a nested list such as
    ``"ap_int<8>, 4"`` keeps ``ap_int<8>`` in one piece.
    """
    parts, current, depth = [], "", 0
    for char in inner:
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def stream_element_type(stream_type: str) -> str:
    """Element type of a stream port type as written in C++.

    ``"hls::stream< int32_t >"`` -> ``"int32_t"``. The optional second template
    argument of ``hls::stream<T, DEPTH>`` is dropped: on the CPU the depth comes
    from the Allo stream declaration, which is the buffer that actually exists.
    """
    open_idx = stream_type.find("<")
    close_idx = stream_type.rfind(">")
    if open_idx < 0 or close_idx < open_idx:
        raise ValueError(f"Not a stream type: {stream_type}")
    args = split_template_args(stream_type[open_idx + 1 : close_idx])
    if not args:
        raise ValueError(f"Stream type has no element type: {stream_type}")
    return args[0]


class _StreamShape:
    """Sentinel shape for an ``hls::stream<T> &`` parameter.

    It is deliberately not sized: a stream is neither a scalar ``()``, an array
    (tuple of dims), nor a pointer ``None``, so shape-dispatching code fails
    loudly rather than silently emitting a bad cast for it.
    """

    def __repr__(self):
        return "STREAM"


STREAM = _StreamShape()


def resolve_nb_type(hls_type: str) -> str:
    """Map an HLS type like ap_int<8> to a nanobind-compatible C type.

    nanobind ndarray<T> requires a concrete arithmetic type; ap_int<N> is not
    recognised by nanobind, so we convert to the equivalent stdint type.
    """
    m = re.match(r"^ap_(u?)int<(\d+)>$", hls_type)
    if m:
        prefix = "u" if m.group(1) == "u" else ""
        bits = int(m.group(2))
        return f"{prefix}int{bits}_t"
    return hls_type


def parse_cpp_function(code, target_function):
    """
    Parse a C++ file to find a specific function and extract its parameter types and shapes.

    Args:
        code (str): The C++ code as a string
        target_function (str): The name of the function to find

    Returns:
        list: A list of tuples containing (type, shape) for each parameter
            - shape is a tuple of dimensions for arrays
            - shape is () for scalars
            - shape is None for pointers
            - shape is STREAM for hls::stream<T> references, in which case the
              type is the stream type as written, e.g. "hls::stream<int8_t>"
    """
    # Function pattern that works for both declarations and definitions
    function_pattern = r"(\w+)\s+" + re.escape(target_function) + r"\s*\((.*?)\)\s*[{;]"

    # Find the function in the code
    function_match = re.search(function_pattern, code, re.DOTALL)
    if not function_match:
        return None

    # Extract return type and parameters
    # return_type = function_match.group(1)
    params_str = function_match.group(2)

    # Drop inline comments: EmitVivadoHLS annotates stream parameters with
    # their depth (`hls::stream<int8_t> &v0 /* v0[2] */`), which would
    # otherwise look like array dimensions.
    params_str = re.sub(r"/\*.*?\*/", " ", params_str, flags=re.DOTALL)

    # Split parameters. Angle brackets are tracked alongside square ones so a
    # comma inside a template argument list does not split a parameter.
    params = []
    current_param = ""
    bracket_count = 0

    for char in params_str:
        if char == "," and bracket_count == 0:
            params.append(current_param.strip())
            current_param = ""
        else:
            current_param += char
            if char in "[<":
                bracket_count += 1
            elif char in "]>":
                bracket_count -= 1

    if current_param.strip():
        params.append(current_param.strip())

    # Process each parameter to extract type and shape.
    # We use _TYPE_TOKEN so that HLS types like ap_int<8> are captured whole.
    # We also added _STREAM_TOKEN to capture streams at the interface
    result = []
    for param in params:
        # Check if parameter is an hls::stream reference. This must be tried
        # before the scalar pattern, whose `\w+` type branch would otherwise
        # match the element type inside the angle brackets.
        stream_pattern = rf"({_STREAM_TOKEN})\s*&\s*(\w+)"
        stream_match = re.search(stream_pattern, param)

        if stream_match:
            result.append((stream_match.group(1), STREAM))
            continue

        # Check if parameter is a pointer
        pointer_pattern = rf"({_TYPE_TOKEN})\s+\*(\w+)"
        pointer_match = re.search(pointer_pattern, param)

        if pointer_match:
            param_type = pointer_match.group(1)
            result.append((param_type, None))
            continue

        # Check if parameter is an array
        array_pattern = rf"({_TYPE_TOKEN})\s+(\w+)((?:\[\d+\])+)"
        array_match = re.search(array_pattern, param)

        if array_match:
            param_type = array_match.group(1)
            array_dims_str = array_match.group(3)

            dims = []
            dim_pattern = r"\[(\d+)\]"
            for dim_match in re.finditer(dim_pattern, array_dims_str):
                dims.append(int(dim_match.group(1)))

            result.append((param_type, tuple(dims)))
            continue

        # Scalar
        scalar_pattern = rf"({_TYPE_TOKEN})\s+(\w+)"
        scalar_match = re.search(scalar_pattern, param)

        if scalar_match:
            param_type = scalar_match.group(1)
            result.append((param_type, ()))

    return result


class IPModule:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        top,
        impl,
        include_paths=None,
        link_hls=True,
        input_idx=None,
        output_idx=None,
    ):
        # ``input_idx`` / ``output_idx`` declare, per argument position, whether
        # the IP *reads* (input) or *writes* (output) that argument. They are
        # only required for arguments the tool cannot direct on its own -- most
        # importantly ``hls::stream<T> &`` ports, which use the same C++ syntax
        # whether the IP reads or writes them (see ``_StreamShape``). They mirror
        # the AIE ``ExternalModule`` API; the IR builder reads ``obj.input_idx``.
        # Default ``None`` preserves the historic memref/scalar behaviour.
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.top = top
        self.impl = os.path.abspath(os.path.expanduser(impl))
        if not os.path.exists(self.impl):
            raise FileNotFoundError(
                f"Path does not exist: {self.impl}. Consider using an absolute path."
            )
        self.abs_path = os.path.dirname(self.impl)
        # Build artifacts (the generated wrapper .cpp, its .o and the .so) go in
        # a private temp directory instead of a `_tmp` folder beside the IP
        # source, so a run leaves nothing behind in the user's tree.
        #
        # Kept as an attribute rather than used in a `with` block on purpose:
        # the directory is created here but consumed later, by
        # `compile_shared_lib()` and `__call__()`, so its lifetime has to be the
        # IPModule's, not this function's. Cleanup therefore happens when the
        # IPModule is collected (or at interpreter exit). That is safe for the
        # JIT: `ExecutionEngine` loads the .so eagerly at construction, and a
        # loaded library stays mapped after its file is unlinked.
        # pylint: disable=consider-using-with
        self._temp_dir = tempfile.TemporaryDirectory(prefix="allo_ip_")
        self.temp_path = self._temp_dir.name
        if include_paths is None:
            include_paths = []
        self.include_paths = include_paths + [self.abs_path]
        if link_hls:
            if os.system("which vitis_hls >> /dev/null") == 0:
                self.include_paths.append(
                    "/".join(os.popen("which vitis_hls").read().split("/")[:-2])
                    + "/include"
                )
            elif os.system("which vivado_hls >> /dev/null") == 0:
                self.include_paths.append(
                    "/".join(os.popen("which vivado_hls").read().split("/")[:-2])
                    + "/include"
                )
            else:
                raise RuntimeError(
                    "Please install Vivado/Vitis HLS and add it to your PATH"
                )

        # Parse signature
        with open(self.impl, "r", encoding="utf-8") as f:
            code = f.read()
            self.args = parse_cpp_function(code, self.top)
        assert self.args is not None, f"Failed to parse {self.impl}"
        self.lib_name = f"py{self.top}_{hash(time.time_ns())}"
        self.c_wrapper_file = os.path.join(self.temp_path, f"{self.lib_name}.cpp")

    @property
    def has_stream_args(self):
        """True if any argument is an ``hls::stream<T> &`` port.

        Such an IP is integrated natively by the HLS targets (vitis_hls /
        vivado_hls), and by the dataflow simulator through the stream shim
        (:meth:`generate_stream_sim_wrapper`). The remaining CPU paths
        reinterpret-cast raw pointers and have no way to represent a FIFO, so
        they refuse it up front.
        """
        return any(shape is STREAM for _, shape in self.args)

    @property
    def stream_arg_indices(self):
        """Positions of the ``hls::stream<T> &`` arguments, in order."""
        return [i for i, (_, shape) in enumerate(self.args) if shape is STREAM]

    def _reject_stream_on_cpu(self):
        if self.has_stream_args:
            raise NotImplementedError(
                f"IP '{self.top}' has hls::stream<T> arguments, which are "
                "supported for the vitis_hls/vivado_hls targets (csyn and "
                "beyond) and for the dataflow simulator "
                "(df.build(..., target='simulator')). They cannot run on the "
                "plain 'llvm' target or in vitis_hls 'csim' mode: those call the "
                "IP once, sequentially, so a blocking stream read would never be "
                "satisfied."
            )

    def generate_nanobind_wrapper(self):
        self._reject_stream_on_cpu()
        out_str = "// Auto-generated by Allo\n\n"
        # Standard headers
        out_str += "#include <cstdint>\n"
        out_str += "#include <iostream>\n"
        out_str += "#include <nanobind/nanobind.h>\n"
        out_str += "#include <nanobind/ndarray.h>\n"
        out_str += f'#include "{os.path.basename(self.impl)}"\n'
        out_str += "\nnamespace nb = nanobind;\n\n"

        # For the nanobind interface we must use concrete arithmetic types;
        # HLS types like ap_int<8> are not nanobind-compatible, so we map them.
        nb_types = [resolve_nb_type(t) for (t, _) in self.args]

        # Function signature using nanobind-compatible types
        out_str += f"void {self.lib_name}(\n"
        for i, ((arg_type, arg_shape), nb_type) in enumerate(zip(self.args, nb_types)):
            if arg_shape is None or len(arg_shape) > 0:
                out_str += f"  const nb::ndarray<{nb_type}> &arg{i}"
            else:
                out_str += f"  {nb_type} arg{i}"
            out_str += ",\n" if i < len(self.args) - 1 else ") {\n"

        # Function body: cast nb types back to the original HLS types
        out_str += "\n"
        in_ptrs = []
        for i, ((arg_type, arg_shape), nb_type) in enumerate(zip(self.args, nb_types)):
            if arg_shape is None or len(arg_shape) == 1:
                out_str += (
                    f"  {arg_type} *p_arg{i} = "
                    f"reinterpret_cast<{arg_type} *>(arg{i}.data());\n"
                )
                in_ptrs.append(f"p_arg{i}")
            elif len(arg_shape) == 0:
                out_str += f"  {arg_type} p_arg{i} = ({arg_type})arg{i};\n"
                in_ptrs.append(f"p_arg{i}")
            else:
                out_str += (
                    f"  {arg_type} *p_arg{i} = "
                    f"reinterpret_cast<{arg_type} *>(arg{i}.data());\n"
                )
                tail_shape = "[" + "][".join([str(s) for s in arg_shape[1:]]) + "]"
                out_str += (
                    f"  {arg_type} (*p_arg{i}_nd){tail_shape} = "
                    f"reinterpret_cast<{arg_type} (*){tail_shape}>(p_arg{i});\n"
                )
                in_ptrs.append(f"p_arg{i}_nd")

        out_str += "\n"
        out_str += f"  {self.top}({', '.join(in_ptrs)});\n"
        out_str += "}\n\n"
        out_str += f"\nNB_MODULE({self.lib_name}, m) {{\n"
        out_str += f'  m.def("{self.top}", &{self.lib_name}, "{self.top} wrapper");\n'
        out_str += "}\n"
        with open(self.c_wrapper_file, "w", encoding="utf-8") as f:
            f.write(out_str)
        return self.c_wrapper_file

    def compile_nanobind(self):
        self.generate_nanobind_wrapper()

        # Get nanobind paths and configuration using Python API
        try:
            nanobind_include = subprocess.check_output(
                ["python3", "-c", "import nanobind; print(nanobind.include_dir())"],
                universal_newlines=True,
            ).strip()

            # Get the nanobind cmake directory to find the static library
            nanobind_cmake_dir = subprocess.check_output(
                ["python3", "-c", "import nanobind; print(nanobind.cmake_dir())"],
                universal_newlines=True,
            ).strip()

            # Get Python include directory
            python_include = subprocess.check_output(
                [
                    "python3",
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('include'))",
                ],
                universal_newlines=True,
            ).strip()

            # Get Python library directory for linking
            python_libdir = subprocess.check_output(
                [
                    "python3",
                    "-c",
                    "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
                ],
                universal_newlines=True,
            ).strip()

            # Get extension suffix
            extension_suffix = subprocess.check_output(
                [
                    "python3",
                    "-c",
                    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))",
                ],
                universal_newlines=True,
            ).strip()

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to get nanobind configuration. Make sure nanobind is installed."
            ) from exc

        # Find the nanobind static library
        # The library is typically in the parent directory of cmake_dir or in a lib subdirectory
        nanobind_base = os.path.dirname(nanobind_cmake_dir)
        possible_lib_paths = [
            os.path.join(nanobind_base, "libnanobind.a"),
            os.path.join(nanobind_base, "lib", "libnanobind.a"),
            os.path.join(nanobind_cmake_dir, "libnanobind.a"),
        ]

        nanobind_lib = None
        for lib_path in possible_lib_paths:
            if os.path.exists(lib_path):
                nanobind_lib = lib_path
                break

        # If static library not found, we need to compile nanobind from source
        if nanobind_lib is None:
            # Get the nanobind source directory
            nanobind_src_dir = subprocess.check_output(
                [
                    "python3",
                    "-c",
                    "import nanobind; import os; print(os.path.dirname(nanobind.__file__))",
                ],
                universal_newlines=True,
            ).strip()

            # Compile nanobind source file
            nanobind_src = os.path.join(nanobind_src_dir, "src", "nb_combined.cpp")
            if not os.path.exists(nanobind_src):
                raise RuntimeError(
                    f"Cannot find nanobind source file at {nanobind_src}. "
                    "Please ensure nanobind is properly installed."
                )

            # nanobind has external dependencies (robin_map) in the ext directory
            nanobind_ext_dir = os.path.join(
                nanobind_src_dir, "ext", "robin_map", "include"
            )
            nanobind_src_include = os.path.join(nanobind_src_dir, "src")

            nanobind_obj = os.path.join(self.temp_path, "nanobind.o")
            compile_nb_cmd = (
                f"g++ -c -std=c++17 -fPIC -fvisibility=hidden "
                f"-I{nanobind_include} -I{python_include} "
                f"-I{nanobind_ext_dir} -I{nanobind_src_include} "
                f"{nanobind_src} -o {nanobind_obj}"
            )
            print(compile_nb_cmd)
            try:
                subprocess.check_output(
                    compile_nb_cmd, shell=True, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile nanobind source: {exc.output.decode() if exc.output else ''}"
                ) from exc

            nanobind_lib = nanobind_obj

        # Build the compilation command
        cmd = f"g++ -shared -std=c++17 -fPIC -fvisibility=hidden -I{nanobind_include} -I{python_include}"
        cmd += " " + " ".join(
            ["-I" + (path if path != "" else ".") for path in self.include_paths]
        )
        srcs = [self.c_wrapper_file]
        cmd += " " + " ".join(srcs)
        cmd += f" {nanobind_lib}"
        cmd += f" -L{python_libdir}"
        cmd += f" -o {self.temp_path}/{self.lib_name}{extension_suffix}"
        print(cmd)
        try:
            subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to compile nanobind wrapper for {self.lib_name}! "
                f"{exc.output.decode() if exc.output else ''}"
            ) from exc

    def generate_mlir_c_wrapper(self):
        self._reject_stream_on_cpu()
        out_str = "// Auto-generated by Allo\n\n"
        # Add headers
        out_str += "#include <iostream>\n"
        out_str += '#include "mlir/ExecutionEngine/CRunnerUtils.h"\n'
        out_str += f'#include "{self.impl}"\n'
        out_str += "\n"
        # Generate function interface
        unranked_memrefs = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if len(arg_shape) > 0:
                unranked_memrefs.append(f"int64_t rank_{i}, void *ptr_{i}")
            else:
                unranked_memrefs.append(f"{arg_type} in{i}")
        unranked_memrefs_str = ", ".join(unranked_memrefs)
        out_str += (
            f'extern "C" {_EXPORT_ATTR} void '
            f"{self.lib_name}({unranked_memrefs_str}) {{\n"
        )
        in_ptrs = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if len(arg_shape) == 0:  # scalar
                in_ptrs.append(f"in{i}")
                continue
            out_str += (
                f"  UnrankedMemRefType<{arg_type}> in{i} = {{rank_{i}, ptr_{i}}};\n"
            )
            out_str += f"  DynamicMemRefType<{arg_type}> ranked_in{i}(in{i});\n"
            out_str += f"  {arg_type} *in{i}_ptr = ({arg_type} *)ranked_in{i}.data;\n"
            if len(arg_shape) == 1:
                in_ptrs.append(f"in{i}_ptr")
            else:
                tail_shape = "[" + "][".join([str(s) for s in arg_shape[1:]]) + "]"
                out_str += f"  {arg_type} (*in{i}_nd){tail_shape} = "
                out_str += f"reinterpret_cast<{arg_type} (*){tail_shape}>(in{i}_ptr);\n"
                in_ptrs.append(f"in{i}_nd")
        # Call library function
        out_str += f"  {self.top}({', '.join(in_ptrs)});\n"
        out_str += "}\n"
        with open(self.c_wrapper_file, "w", encoding="utf-8") as f:
            f.write(out_str)
        return self.c_wrapper_file

    def generate_stream_sim_wrapper(self):
        """Emit the C++ wrapper that runs a stream IP under the CPU simulator.

        The wrapper is the bridge between two worlds:

        * Allo's JIT-compiled module, which represents each stream as a ring
          buffer -- three ``memref``s (data, head, tail) that the MLIR->LLVM ABI
          hands over as *unranked memref descriptors*, i.e. one
          ``(int64_t rank, void *descriptor)`` pair per memref.
        * the IP, compiled against Allo's shim ``hls::stream<T>`` (see
          ``allo/backend/ip_sim/hls_stream.h``), which needs an
          ``AlloFifo<T>``: a data pointer, a capacity and the two index
          pointers.

        So the body reconstructs each memref with ``DynamicMemRefType`` (exactly
        as :meth:`generate_mlir_c_wrapper` does for array arguments -- that
        class knows the descriptor layout, so we never match fields by hand),
        fills in one ``AlloFifo`` per stream, wraps each in a shim
        ``hls::stream`` and calls the IP. Nothing is copied: the IP reads and
        writes Allo's buffers in place.

        Non-stream arguments are passed the same way
        :meth:`generate_mlir_c_wrapper` passes them, so an IP may mix array,
        scalar and stream ports.
        """
        if not self.has_stream_args:
            raise ValueError(
                f"IP '{self.top}' has no hls::stream<T> arguments; use "
                "generate_mlir_c_wrapper() instead."
            )
        out_str = "// Auto-generated by Allo (dataflow simulator stream shim)\n\n"
        out_str += "#include <cassert>\n"
        out_str += "#include <cstdint>\n"
        # The shim must be seen before the IP source. Allo puts its own
        # directory first on the include path, so this resolves to
        # allo/backend/ip_sim/hls_stream.h, not to Vitis's header.
        out_str += "#include <hls_stream.h>\n"
        out_str += '#include "mlir/ExecutionEngine/CRunnerUtils.h"\n'
        out_str += f'#include "{self.impl}"\n'
        out_str += "\n"

        # --- entry signature -------------------------------------------------
        params = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if arg_shape is STREAM:
                # One unranked-memref pair per ring-buffer field.
                for field in ("data", "head", "tail"):
                    params.append(f"int64_t s{i}_{field}_rank, void *s{i}_{field}_ptr")
            elif arg_shape is None or len(arg_shape) > 0:
                params.append(f"int64_t rank_{i}, void *ptr_{i}")
            else:
                params.append(f"{arg_type} in{i}")
        out_str += (
            f'extern "C" {_EXPORT_ATTR} void '
            f'{self.lib_name}({", ".join(params)}) {{\n'
        )

        # --- body ------------------------------------------------------------
        in_args = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if arg_shape is STREAM:
                elem_type = stream_element_type(arg_type)
                # head / tail are always `memref<i32>` on the Allo side.
                for field, field_type in (
                    ("data", elem_type),
                    ("head", "int32_t"),
                    ("tail", "int32_t"),
                ):
                    out_str += (
                        f"  UnrankedMemRefType<{field_type}> s{i}_{field}_u = "
                        f"{{s{i}_{field}_rank, s{i}_{field}_ptr}};\n"
                    )
                    out_str += (
                        f"  DynamicMemRefType<{field_type}> s{i}_{field}"
                        f"(s{i}_{field}_u);\n"
                    )
                out_str += (
                    f"  assert(s{i}_data.rank == 1 && "
                    '"Allo FIFO storage must be a 1-D memref");\n'
                )
                out_str += (
                    f"  assert(s{i}_data.strides[0] == 1 && "
                    '"Allo FIFO storage must be contiguous");\n'
                )
                out_str += f"  AlloFifo<{elem_type}> s{i}_fifo;\n"
                out_str += f"  s{i}_fifo.data = s{i}_data.data + s{i}_data.offset;\n"
                # The number of slots is the size of the data memref, which the
                # simulator allocates as `depth + 1`.
                out_str += f"  s{i}_fifo.cap = (int32_t)s{i}_data.sizes[0];\n"
                out_str += f"  s{i}_fifo.head = s{i}_head.data + s{i}_head.offset;\n"
                out_str += f"  s{i}_fifo.tail = s{i}_tail.data + s{i}_tail.offset;\n"
                out_str += f"  hls::stream<{elem_type}> s{i}(&s{i}_fifo);\n"
                in_args.append(f"s{i}")
                continue
            if arg_shape is not None and len(arg_shape) == 0:  # scalar
                in_args.append(f"in{i}")
                continue
            out_str += (
                f"  UnrankedMemRefType<{arg_type}> in{i} = {{rank_{i}, ptr_{i}}};\n"
            )
            out_str += f"  DynamicMemRefType<{arg_type}> ranked_in{i}(in{i});\n"
            out_str += f"  {arg_type} *in{i}_ptr = ({arg_type} *)ranked_in{i}.data;\n"
            if arg_shape is None or len(arg_shape) == 1:
                in_args.append(f"in{i}_ptr")
            else:
                tail_shape = "[" + "][".join([str(s) for s in arg_shape[1:]]) + "]"
                out_str += f"  {arg_type} (*in{i}_nd){tail_shape} = "
                out_str += f"reinterpret_cast<{arg_type} (*){tail_shape}>(in{i}_ptr);\n"
                in_args.append(f"in{i}_nd")
        out_str += f"  {self.top}({', '.join(in_args)});\n"
        out_str += "}\n"
        with open(self.c_wrapper_file, "w", encoding="utf-8") as f:
            f.write(out_str)
        return self.c_wrapper_file

    def compile_shared_lib(self, stream_sim=False):
        """Compile the IP into a .so the JIT can call.

        ``stream_sim=True`` selects the dataflow-simulator flavour: the IP is
        built against Allo's shim ``hls::stream`` (whose directory therefore goes
        first on the include path) and entered through
        :meth:`generate_stream_sim_wrapper`.
        """
        # Used in direct function call in an Allo kernel
        if stream_sim:
            self.generate_stream_sim_wrapper()
        else:
            self.generate_mlir_c_wrapper()
        if os.system("which llvm-config >> /dev/null") != 0:
            raise RuntimeError("Please install LLVM and add it to your PATH")
        # -fvisibility=hidden: export only the wrapper entry (which carries an
        # explicit visibility("default") attribute), so an IP's own top function
        # cannot be interposed by a same-named one from another IP's .so loaded
        # into the same process. See `_EXPORT_ATTR`.
        cmd = "g++ -c -std=c++14 -fpic -fvisibility=hidden "
        # suppose the build directory is under llvm-project
        include_paths = list(self.include_paths) + [
            "/".join(os.popen("which llvm-config").read().split("/")[:-3])
            + "/mlir/include"
        ]
        if stream_sim:
            # First, so that `#include <hls_stream.h>` in the IP finds Allo's
            # shim and not Vitis's header (which `link_hls=True` may also have
            # put on this list).
            include_paths.insert(0, IP_SIM_INCLUDE_DIR)
            # The IP body keeps its `#pragma HLS ...` lines, which g++ does not
            # know; they are hardware directives and irrelevant on the CPU.
            cmd += "-Wno-unknown-pragmas "
            if os.getenv("ALLO_IP_SIM_OPENMP") == "1":
                # Opt-in: compiles the `omp taskyield` hint in the shim's spin
                # loop. Off by default because g++ links libgomp while the
                # simulator's JIT-compiled code uses LLVM's libomp, and hosting
                # two OpenMP runtimes in one process is unsafe. The hint has no
                # effect on correctness -- `usleep(1)` does the yielding.
                cmd += "-fopenmp "
        cmd += " ".join(
            ["-I" + (path if path != "" else ".") for path in include_paths]
        )
        srcs = [self.c_wrapper_file]
        obj_files = []
        for src in srcs:
            subcmd = cmd
            subcmd += " " + src
            obj = f"{self.temp_path}/{src.split('/')[-1]}.o"
            subcmd += " -o " + obj
            print(subcmd)
            try:
                subprocess.check_output(subcmd, shell=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile {src.split('/')[-1]}.o!"
                ) from exc
            obj_files.append(obj)
        # Name the .so after lib_name (which carries a per-instance hash), not
        # self.top: two IPModules wrapping the same top function would otherwise
        # both write lib<top>.so, and the second overwrites the first -- so the
        # first module's JIT can no longer find its (uniquely-named) symbol when
        # both live in one process (e.g. two tests in one pytest run).
        so_path = f"{self.temp_path}/lib{self.lib_name}.so"
        link_flags = ""
        if stream_sim and os.getenv("ALLO_IP_SIM_OPENMP") == "1":
            link_flags = "-fopenmp "
        cmd = f"g++ -shared {link_flags}-o {so_path} " + " ".join(obj_files)
        print(cmd)
        try:
            subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to compile {so_path}!") from exc
        return so_path

    def __call__(self, *args):
        self.compile_nanobind()
        sys.path.append(self.temp_path)
        self.lib = importlib.import_module(f"{self.lib_name}")
        return getattr(self.lib, f"{self.top}")(*args)

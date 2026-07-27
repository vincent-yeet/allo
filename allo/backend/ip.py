# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import re
import importlib
import subprocess
import time

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
        self.temp_path = os.path.join(self.abs_path, "_tmp")
        os.makedirs(self.temp_path, exist_ok=True)
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

        Such an IP can only be integrated for the HLS targets (vitis_hls /
        vivado_hls). The CPU paths below reinterpret-cast raw pointers and have
        no way to represent a stream, so they refuse it up front.
        """
        return any(shape is STREAM for _, shape in self.args)

    def _reject_stream_on_cpu(self):
        if self.has_stream_args:
            raise NotImplementedError(
                f"IP '{self.top}' has hls::stream<T> arguments, which are only "
                "supported for the vitis_hls/vivado_hls targets (csyn and "
                "beyond). They cannot run on the CPU 'llvm'/'simulator' targets "
                "or in vitis_hls 'csim' mode."
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
        out_str += f'extern "C" void {self.lib_name}({unranked_memrefs_str}) {{\n'
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

    def compile_shared_lib(self):
        # Used in direct function call in an Allo kernel
        self.generate_mlir_c_wrapper()
        if os.system("which llvm-config >> /dev/null") != 0:
            raise RuntimeError("Please install LLVM and add it to your PATH")
        cmd = "g++ -c -std=c++14 -fpic "
        # suppose the build directory is under llvm-project
        self.include_paths.append(
            "/".join(os.popen("which llvm-config").read().split("/")[:-3])
            + "/mlir/include"
        )
        cmd += " ".join(
            ["-I" + (path if path != "" else ".") for path in self.include_paths]
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
        cmd = f"g++ -shared -o {so_path} " + " ".join(obj_files)
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

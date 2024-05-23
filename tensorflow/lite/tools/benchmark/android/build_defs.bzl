# The get_shared_libs macro generates two lists (outputs and deps) to add all shared libraries from the shared_libs folder:
# The first list (outputs) contains library name and library path for cc_import
# The second list (deps) contains only library names to define extra dependendences
# If there is no shared libraries in the shared_libs folder a warning message is generated and both lists are empty.

def get_shared_libs():
    outputs = []
    deps=[]
    libs = native.glob(["shared_libs/*.so"])
    if(libs):
        for lib in libs:
            lib_name = lib.replace("shared_libs/","")
            outputs.append((lib_name, lib))
            deps.append(lib_name)
    else:
        print(
            "\n\033[1;33mWARNING:\033[0m There is no shared library in the shared_libs folder. " +
            "If you want to add a delegate library to your apk, please build it and copy it to the shared_libs folder."
        )
    return outputs, deps
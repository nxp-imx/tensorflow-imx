# The get_delegate_lib macro for getting delegate library from the shared_libs folder  or generating error.

def get_delegate_lib():
    libs = native.glob(["shared_libs/lib*_delegate.so"])
    if(libs):
        if(len(libs) > 1):
            print(
                "\n\033[1;33mWARNING:\033[0m There are more delegate libraries in the shared_libs folder. " +
                "Please delete all unused delegate libraries and keep only one. " +
                "\n\033[1;33mWARNING:\033[0m The " + libs[0] + " library was added to the benchmark_model application by default."
            )
        return libs[0]
    else:
        fail(msg="There is no delegate library in shared_libs folder. " +
            "Please build one and copy it to the shared_libs folder."
        )
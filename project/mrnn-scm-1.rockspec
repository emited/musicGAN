package = "mrnn"
version = "scm-1"

source = {
   url = "https://bitbucket.org/emited/mrnn.git",
}

description = {
   summary = "music generation",
   detailed = [[
   ]],
   homepage = "https://bitbucket.org/emited/mrnn.git",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && echo $(PREFIX) && $(MAKE) install"
}

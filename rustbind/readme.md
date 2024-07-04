# How to solve the include errors in "interfaces"

You should first "make install" troy to somewhere, so that it produces the includes folder. You can specify where to install by "export DESTDIR=...". Then add to your intellisence this folder.
For clangd, you can add this line to your ".clangd" file:
```
CompileFlags:
  Add: 
    - -I/data/lxq/troy-nova/build/install/usr/local/include
```
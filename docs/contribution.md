# Contribution Guidelines

## I. Coding convention

DaisyKit follows [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with some exceptions:

- Source code file names: `.cpp` for source file and `.h` for header files.
- Accept `using namespace` in `.cpp` files.

Coding convention for the code base should be formatted and checked by [clang-format](https://clang.llvm.org/docs/ClangFormat.html). Configuration file `.clang-format` for code formatting can be found [here](https://github.com/VNOpenAI/daisykit/blob/master/.clang-format).

**Setup for Visual Studio Code (VS Code)**

VS Code can use clang-format to format source code file (Ctrl+Shift+i).

- Install Clang-Format extension in VS Code: <https://marketplace.visualstudio.com/items?itemName=xaver.clang-format>.
- Configure `C_Cpp:Clang_format_style` as following:

```
{BasedOnStyle: Google, PointerAlignment: Left, IncludeBlocks: Preserve, DerivePointerAlignment: false}
```

![Configure Clang Format for VS Code - DaisyKit project](config-clang-format-vscode.png)

## II. Contribute to DaisyKit SDK

Create a pull request to <https://github.com/VNOpenAI/daisykit>. Visit repository for the setup instruction.

**Next tasks:** Build model training code, inference code, design and build flow architecture, write documentation and tutorials.

## III. Contribute to DaisyKit Android

Create a pull request to <https://github.com/VNOpenAI/daisykit-android>. Visit repository for the setup instruction.

**Next tasks:** Build wrappers for Kotlin, Java and example applications.

## IV. Contribute to Daisykit iOS

Create a pull request to <https://github.com/VNOpenAI/daisykit-ios>. Visit repository for the setup instruction.

**Next tasks:** Build base app, wrappers for Swift, Objective-C and example applications.

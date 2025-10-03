# V1 – KLT Feature Tracker

This version implements the **Kanade–Lucas–Tomasi (KLT) feature tracker**, a computer vision algorithm used to detect and track features in images across frames.
It builds a static library "libklt.a" and several example programs `example1`–`example5` demonstrating feature detection, tracking, and visualization.

---

## Project Structure

* `*.c` and `*.h`: Core source files implementing KLT (e.g., `trackFeatures.c`, `selectGoodFeatures.c`, etc.)
* `libklt.a`: Static library built from the source files
* `example1.c` … `example5.c`: Example programs showing how to use the KLT tracker
* `Makefile`: Build automation

---

## Build Instructions

### Prerequisites

* GCC (GNU Compiler Collection)
* Standard C libraries

### Compile Everything

```bash
make all
```

This will:

* Build the static library `libklt.a`
* Compile all five example executables (`example1` … `example5`)

### Clean Build Files

```bash
make clean
```

---

## Running Examples

Each example can be executed directly after building. For example:

```bash
./example1
```

## Notes

* `-DNDEBUG` disables debugging assertions. Remove it from the `Makefile` if you want debug mode.
* Profiling is enabled with `-pg`. Use `gprof` to analyze performance after running examples.
* If you encounter issues with sorting performance, you can uncomment `-DKLT_USE_QSORT` in the `Makefile` to force use of the standard C `qsort()`.


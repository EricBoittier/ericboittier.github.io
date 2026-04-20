---
title: Fortran is fast. Profile your code to make it faster!
subtitle: Profiling Fortran code with gprof

# Summary for listings and search engines
summary: Fortan go "brrr"... but can we make it go faster? Profiling code is the best way to improve efficiency. Here you'll find a short explanation on how to do this in Fortran.

# Link this post with a project
projects: []

# Date published
date: "2022-08-17T00:00:00Z"

# Date updated
date: "2022-08-17T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: true

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Image credit: [**Boardgamegeek**](https://boardgamegeek.com/image/78330/fortran)'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

tags:
- code
- fortran

categories:
- optimization
- programming
---

## All About Fortran
### What is Fortran?

Fortran is a dinosaur code language. It was released around 65 years ago (date of writing) by John Backus and IBM for electric computing machines powered by the fossilized remains of dinosaurs who lived around 65 million years ago. 

'Too lazy' to write assembly, Backus wrote this compiled imperative language to save himself time when composing complicated mathematical formulas, giving rise to the 'Formula Translation' language or FORTRAN. Nowadays, FORTRAN is considered too verbose by modern programming standards. Although much slower, Python might be considered the new Formula Translation language. Routines like the Fast Fourier Transform have been reduced to one line (np.fft()), where the number of lines of pure FORTRAN and assembly code needed are around 1 to 2 orders of magnitude longer, respectively.

### Why Fortran?

Fortran is fast. Many legacy applications rely on Fortran for reasons related to speed and compatibility. 

## The Good Type of Profiling
### How can we test the speed of our code?

Profiling is a strategy to monitor the performance of one's code.
Results often show the time spent in individual routines, the number of calls, as well as the order in which routines have been accessed. 

## Profiling Fortran with gprof

Fortran must first be compiled with the following, additional flags:
```bash
... -pg fcheck=bounds,mem
```
The 'fcheck' option allows the compiler to produce warnings for attempts at accessing undefined memory, etc.

Once compiled, run your code as usual:
```bash
./yourcode 
```
A file named 'gmon.out' will be created in the current directory.
To see the results of the profiling
```bash
gprof -l -b ./yourcode > gprof.log
```
The -g and -l flags in the compilation and profiling steps, respectively, allow for a line by line analysis of time spent during computation. Without these options, the profiler will show total time spent in each subroutine.











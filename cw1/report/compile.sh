#!/bin/bash
pdflatex cw1-report.tex
bibtex cw1-report.aux
pdflatex cw1-report.tex
pdflatex cw1-report.tex


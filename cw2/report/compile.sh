#!/bin/bash
pdflatex cw2-report.tex
bibtex cw2-report.aux
pdflatex cw2-report.tex
pdflatex cw2-report.tex
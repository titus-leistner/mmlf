PDF = figure.pdf
OUT = output
TEX = $(PDF:.pdf=.tex)
AUX = $(PDF:.pdf=.aux)
SVG = $(PDF:.pdf=.svg)

all: $(PDF)

svg: $(SVG)

$(PDF): $(TEX)
	pdflatex -halt-on-error -output-directory=$(OUT) $(TEX)

$(SVG): $(PDF)
	pdf2svg $(OUT)/$(PDF) $(OUT)/$(SVG)

view: all
	zathura --fork $(OUT)/$(PDF) && sleep 10 && ls *.tex *.bib | entr make

clean:
	rm $(OUT)/*

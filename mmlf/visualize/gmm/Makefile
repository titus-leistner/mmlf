PDF = figure.pdf
OUT = output
TEX = $(PDF:.pdf=.tex)
AUX = $(PDF:.pdf=.aux)
SVG = $(PDF:.pdf=.svg)
PNG = $(PDF:.pdf=.png)

all: $(PDF)
svg: $(SVG)
png: $(PNG)

$(PDF): $(TEX)
	pdflatex -halt-on-error -output-directory=$(OUT) $(TEX)

$(SVG): $(PDF)
	pdf2svg $(OUT)/$(PDF) $(OUT)/$(SVG)

$(PNG): $(PDF)
	convert -density 300 $(OUT)/$(PDF) -flatten -quality 90 $(OUT)/$(PNG)

view: all
	zathura --fork $(OUT)/$(PDF) && sleep 10 && ls *.tex *.bib | entr make

clean:
	rm $(OUT)/*

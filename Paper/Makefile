LATEXMK = latexmk
MAIN = main

.PHONY: all clean distclean watch

all:
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex

distclean:
	$(LATEXMK) -C $(MAIN).tex
	rm -rf build

# Generate summary table.
results.txt : testzipf.py isles.dat abyss.dat last.dat
	python $< *.dat > $@


# Count words.
.PHONY : dats
dats : isles.dat abyss.dat last.dat

# Count words.
#isles.dat : books/isles.txt
#	python countwords.py $< $@

#abyss.dat : books/abyss.txt
#	python countwords.py $< $@

%.dat : books/%.txt countwords.py
	python countwords.py $< $*.dat 

.PHONY : clean
clean :
	rm -f *.dat

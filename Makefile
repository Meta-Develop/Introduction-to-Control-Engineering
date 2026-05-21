.PHONY: ja en ja-dvi en-dvi clean

ja: ja-dvi
	mkdir -p pdf
	cd ja && rm -f ../build/ja/main.pdf && OSFONTDIR=$$HOME/.local/share/fonts/haranoaji dvipdfmx -f ../common/style/dvipdfmx-haranoaji.map -o ../build/ja/main.pdf ../build/ja/main.dvi
	cp build/ja/main.pdf pdf/control-engineering-ja.pdf

en: en-dvi
	mkdir -p pdf
	cd en && rm -f ../build/en/main.pdf && OSFONTDIR=$$HOME/.local/share/fonts/haranoaji dvipdfmx -f ../common/style/dvipdfmx-haranoaji.map -o ../build/en/main.pdf ../build/en/main.dvi
	cp build/en/main.pdf pdf/control-engineering-en.pdf

ja-dvi:
	mkdir -p build/ja
	rm -f build/ja/main.*
	cd ja && uplatex -interaction=nonstopmode -file-line-error -output-directory=../build/ja main.tex
	cd ja && uplatex -interaction=nonstopmode -file-line-error -output-directory=../build/ja main.tex
	cd ja && uplatex -interaction=nonstopmode -file-line-error -output-directory=../build/ja main.tex

en-dvi:
	mkdir -p build/en
	cd en && uplatex -interaction=nonstopmode -file-line-error -output-directory=../build/en main.tex
	cd en && uplatex -interaction=nonstopmode -file-line-error -output-directory=../build/en main.tex

clean:
	rm -rf build

# Make sure you have the following things installed:
#
#   - Entangled (the daemon)
#   - entangled-filters (the pandoc filters: pip install ...)
#   - Pandoc
#   - BrowserSync (npm install -g ...)
#   - InotifyTools (available from most GNU/Linux distributions)
#
# The website will be built in `./$(docs_dir)`, from which it can be served as
# github.io pages.
#
#
# You should list the Markdown sources here in the order that they should
# appear.
lit_dir             := lit
docs_dir            := docs
assets_dir          := assets
bootstrap_dir       := bootstrap
input_files         := $(lit_dir)/*.md
pandoc_intermediary := $(docs_dir)/pandoc.json

# Arguments to Pandoc
pandoc_args += --from markdown+multiline_tables+simple_tables
pandoc_args += --standalone
pandoc_args += --filter pandoc-doctest
pandoc_args += --filter pandoc-csv2table

# Load syntax definitions for languages that are not supported
# by default. These XML files are in the format of the Kate editor.
# FIXME: unfortunately doesn't work in the Docker container,
#        missing 'language.dtd' schema(?) file (from DOCTYPE)
#        which is mysteriously picked up automagically in
#        the local environment.
# pandoc_args += --syntax-definition $(bootstrap_dir)/elm.xml
# pandoc_args += --syntax-definition $(bootstrap_dir)/pure.xml
pandoc_args += --highlight-style tango

# Arguments to Pandoc specific to HTML5 output
pandoc_args_html += --to html5
pandoc_args_html += --mathjax
pandoc_args_html += --toc
pandoc_args_html += --toc-depth 2
pandoc_args_html += --template $(bootstrap_dir)/template.html
pandoc_args_html += --css css/mods.css
# FIXME: bootstrap after doctest leads to cell output outside of foldable blocks
pandoc_args_html += --filter pandoc-bootstrap

# Arguments to Pandoc specific to PDF output
pandoc_args_pdf += --to pdf
# pandoc_args_pdf += --pdf-engine xelatex
pandoc_args_pdf += --template $(assets_dir)/template.tex
pandoc_args_pdf += --variable documentclass=article
pandoc_args_pdf += --variable geometry="margin=3cm"
# pandoc_args_pdf += --variable mainfont=libertine
# pandoc_args_pdf += --variable sansfont=libertine

# Any file in the `$(lit_dir)` directory that is not a Markdown source
# is to be copied to the `$(docs_dir)` directory
static_files := $(shell find -L $(lit_dir) -type f -not -name '*.md')
static_targets := $(static_files:$(lit_dir)/%=$(docs_dir)/%)

.PHONY: all site clean watch watch-pandoc watch-browser-sync

all: site $(docs_dir)/report.pdf

# This should build everything needed to generate your web site. That includes
# possible Javascript targets that may need compiling.
site: $(docs_dir)/index.html $(docs_dir)/css/mods.css $(static_targets)

clean:
	rm -rf $(docs_dir)

# Starts a tmux with Entangled, Browser-sync and an Inotify loop for running
# Pandoc.
watch:
	@tmux new-session make --no-print-directory watch-pandoc \; \
		split-window -v make --no-print-directory watch-browser-sync \; \
		split-window -v entangled daemon \; \
		select-layout even-vertical \;

watch-pandoc:
	@while true; do \
		inotifywait -e close_write $(bootstrap_dir) $(lit_dir) Makefile; \
		make site; \
	done

watch-browser-sync:
	browser-sync start -w -s $(docs_dir)

$(docs_dir):
	mkdir -p $(docs_dir)

$(pandoc_intermediary): $(input_files) $(docs_dir)
	JUPYTER=1 pandoc $(pandoc_args) $(input_files) -o $@

$(docs_dir)/index.html: Makefile $(pandoc_intermediary) $(bootstrap_dir)/template.html
	pandoc $(pandoc_args_html) $(pandoc_intermediary) -o $@

$(docs_dir)/report.pdf: Makefile $(pandoc_intermediary) $(assets_dir)/template.tex
#	Hack to allow ![](graph.svg) both in the PDF and the website
	ln -sf $(docs_dir)/graph.svg graph.svg
	pandoc $(pandoc_args_pdf) $(pandoc_intermediary) -o $@

$(docs_dir)/css/mods.css: $(bootstrap_dir)/mods.css
	@mkdir -p $(docs_dir)/css
	cp $< $@

$(static_targets): $(docs_dir)/%: $(lit_dir)/%
	@mkdir -p $(dir $@)
	cp $< $@

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
lit_dir     := lit
docs_dir    := docs
input_files := $(lit_dir)/*.md

# Arguments to Pandoc; these are reasonable defaults
pandoc_args += --template bootstrap/template.html
pandoc_args += --css css/mods.css
pandoc_args += -t html5 -s --mathjax --toc
pandoc_args += --toc-depth 1
pandoc_args += --filter pandoc-bootstrap
pandoc_args += --filter pandoc-doctest
pandoc_args += -f markdown+multiline_tables+simple_tables

# Load syntax definitions for languages that are not supported
# by default. These XML files are in the format of the Kate editor.
# FIXME: unfortunately doesn't work in the Docker container,
#        missing 'language.dtd' schema(?) file (from DOCTYPE)
#        which is mysteriously picked up automagically in
#        the local environment.
# pandoc_args += --syntax-definition bootstrap/elm.xml
# pandoc_args += --syntax-definition bootstrap/pure.xml
pandoc_args += --highlight-style tango

# Any file in the `$(lit_dir)` directory that is not a Markdown source
# is to be copied to the `$(docs_dir)` directory
static_files := $(shell find -L $(lit_dir) -type f -not -name '*.md')
static_targets := $(static_files:$(lit_dir)/%=$(docs_dir)/%)

.PHONY: site clean watch watch-pandoc watch-browser-sync

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
		inotifywait -e close_write bootstrap $(lit_dir) Makefile; \
		make site; \
	done

watch-browser-sync:
	browser-sync start -w -s $(docs_dir)

$(docs_dir)/index.html: $(input_files) Makefile
	@mkdir -p $(docs_dir)
	JUPYTER=1 pandoc $(pandoc_args) $(input_files) -o $@

$(docs_dir)/css/mods.css: bootstrap/mods.css
	@mkdir -p $(docs_dir)/css
	cp $< $@

$(static_targets): $(docs_dir)/%: $(lit_dir)/%
	@mkdir -p $(dir $@)
	cp $< $@

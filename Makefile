EGG_INFO := birsvd.egg-info
BUILD_DIR := build
DIST_DIR := $(BUILD_DIR)/dist

.PHONY: clean build

all:

build:
	python -m build --outdir $(DIST_DIR)
	python -m build --sdist --outdir $(DIST_DIR)

clean:
	rm -rf $(BUILD_DIR) $(EGG_INFO) $(OBJECT)

# ROUGE TITAN — RTL verification and software test entry points.
VERILATOR ?= verilator
PYTHON    ?= python3
BUILD_DIR := build
VFLAGS    := --binary --timing --assert -Wall
RTL_MODULES := rtl_mac_array titan_mxu_tile

.PHONY: all test test-mac test-mac-random test-mxu test-mxu-int4 lint pytest cosim synth bench clean

all: lint test pytest

# ---------------------------------------------------------------- RTL simulation (self-checking)
test: test-mac test-mac-random test-mxu test-mxu-int4

test-mac: | $(BUILD_DIR)
	$(VERILATOR) $(VFLAGS) --top-module tb_rtl_mac_array --Mdir $(BUILD_DIR)/mac rtl/rtl_mac_array.sv tb/tb_rtl_mac_array.sv
	./$(BUILD_DIR)/mac/Vtb_rtl_mac_array

test-mac-random: | $(BUILD_DIR)
	$(VERILATOR) $(VFLAGS) --top-module tb_rtl_mac_array_random --Mdir $(BUILD_DIR)/mac_rand rtl/rtl_mac_array.sv tb/tb_rtl_mac_array_random.sv
	./$(BUILD_DIR)/mac_rand/Vtb_rtl_mac_array_random

test-mxu: | $(BUILD_DIR)
	$(VERILATOR) $(VFLAGS) --top-module tb_titan_mxu_tile --Mdir $(BUILD_DIR)/mxu rtl/titan_mxu_tile.sv tb/tb_titan_mxu_tile.sv
	./$(BUILD_DIR)/mxu/Vtb_titan_mxu_tile

test-mxu-int4: | $(BUILD_DIR)
	$(VERILATOR) $(VFLAGS) -GDATA_W=4 -GROWS=8 -GCOLS=8 --top-module tb_titan_mxu_tile --Mdir $(BUILD_DIR)/mxu4 rtl/titan_mxu_tile.sv tb/tb_titan_mxu_tile.sv
	./$(BUILD_DIR)/mxu4/Vtb_titan_mxu_tile

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ---------------------------------------------------------------- lint (warnings are errors)
lint:
	@for m in $(RTL_MODULES); do echo "lint $$m"; $(VERILATOR) --lint-only -Wall --top-module $$m rtl/$$m.sv || exit 1; done
	$(VERILATOR) --lint-only -Wall --timing --top-module tb_titan_mxu_tile rtl/titan_mxu_tile.sv tb/tb_titan_mxu_tile.sv
	$(VERILATOR) --lint-only -Wall --timing --top-module tb_rtl_mac_array_random rtl/rtl_mac_array.sv tb/tb_rtl_mac_array_random.sv

# ---------------------------------------------------------------- software
pytest:
	$(PYTHON) -m pytest -q

cosim:
	$(PYTHON) -m pytest -q -m rtl

synth:
	$(PYTHON) -m titan synth --out reports/synth

bench:
	$(PYTHON) -m titan bench --suite default --out reports

clean:
	rm -rf $(BUILD_DIR)

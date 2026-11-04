VERILATOR ?= verilator
BUILD_DIR := build
RTL := rtl/rtl_mac_array.sv
TB := tb/tb_rtl_mac_array.sv

.PHONY: test lint clean

test:
	$(VERILATOR) --binary --timing --assert -Wall -Wno-fatal --top-module tb_rtl_mac_array --Mdir $(BUILD_DIR) $(RTL) $(TB)
	./$(BUILD_DIR)/Vtb_rtl_mac_array

lint:
	$(VERILATOR) --lint-only -Wall -Wno-fatal --top-module rtl_mac_array $(RTL)

clean:
	rm -rf $(BUILD_DIR)

# GNU Make workspace makefile autogenerated by Premake

ifndef config
  config=release
endif

ifndef verbose
  SILENT = @
endif

ifeq ($(config),release)
  CluE_config = release
  BICO_Experiments_config = release
  BICO_Quickstart_config = release
endif

PROJECTS := CluE BICO_Experiments BICO_Quickstart

.PHONY: all clean help $(PROJECTS) 

all: $(PROJECTS)

CluE:
ifneq (,$(CluE_config))
	@echo "==== Building CluE ($(CluE_config)) ===="
	@${MAKE} --no-print-directory -C . -f CluE.make config=$(CluE_config)
endif

BICO_Experiments: CluE
ifneq (,$(BICO_Experiments_config))
	@echo "==== Building BICO_Experiments ($(BICO_Experiments_config)) ===="
	@${MAKE} --no-print-directory -C . -f BICO_Experiments.make config=$(BICO_Experiments_config)
endif

BICO_Quickstart: CluE
ifneq (,$(BICO_Quickstart_config))
	@echo "==== Building BICO_Quickstart ($(BICO_Quickstart_config)) ===="
	@${MAKE} --no-print-directory -C . -f BICO_Quickstart.make config=$(BICO_Quickstart_config)
endif

clean:
	@${MAKE} --no-print-directory -C . -f CluE.make clean
	@${MAKE} --no-print-directory -C . -f BICO_Experiments.make clean
	@${MAKE} --no-print-directory -C . -f BICO_Quickstart.make clean

help:
	@echo "Usage: make [config=name] [target]"
	@echo ""
	@echo "CONFIGURATIONS:"
	@echo "  release"
	@echo ""
	@echo "TARGETS:"
	@echo "   all (default)"
	@echo "   clean"
	@echo "   CluE"
	@echo "   BICO_Experiments"
	@echo "   BICO_Quickstart"
	@echo ""
	@echo "For more information, see https://github.com/premake/premake-core/wiki"
SHELL    =/bin/bash

# SD ANALYSIS COMPILATION SETTINGS
$(if $(wildcard $(SDDIR)),,\
	$(error The SDDIR ($(SDDIR)) variable is not properly set))
include $(SDDIR)/makefileset.mk

TRIG=$(shell pwd)
include $(TRIG)/trig.mk

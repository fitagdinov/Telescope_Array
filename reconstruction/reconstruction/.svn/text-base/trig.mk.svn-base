
INCS += -I$(TRIG)/inc
trig_libs = -L$(SDLIBDIR) -lsduti $(DSTLIBS)

trig_binlist := sdtrgbk.run remhytrig.run remnotrig.run
trig_binlist += trigp.run trigt.run cntmsd.run pnotrig.run
trig_binfiles = $(addprefix $(SDBINDIR)/, $(trig_binlist))
trig_obj_list = sdtrgbk.o sdtrgbk_util.o sdtrgbkAnalysis.o sdinfo_class.o
trig_obj = $(addprefix $(TRIG)/src/, $(trig_obj_list))

SDBINS += $(trig_binfiles)

.PHONY: trig
trig: $(trig_binfiles)
all:: trig

$(SDBINDIR)/sdtrgbk.run:  $(trig_obj) ; \
$(LD) $(LDFLAGS) -o $@ $^ $(trig_libs)

trig_binfiles_rem=$(filter-out $(SDBINDIR)/sdtrgbk.run, $(trig_binfiles))

$(trig_binfiles_rem): $(SDBINDIR)/%.run : $(TRIG)/src/%.o ; \
$(LD) $(LDFLAGS) -o $@ $^ $(trig_libs)

clean:: ; \
rm -f $(TRIG)/src/*.o $(TRIG)/src/*~ $(TRIG)/*~

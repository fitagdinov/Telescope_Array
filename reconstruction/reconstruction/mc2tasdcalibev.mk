
INCS += -I$(MC2TASDCALIBEV)/inc
mc2tasdcalibev_libs = -L$(SDLIBDIR) -lsdfdrt -lsduti $(ROOTLIBS_ALT) $(DSTLIBS)
mc2tasdcalibev_miscblist = sdmc_split.run sdmc_rt2dst.run
mc2tasdcalibev_miscbins=$(addprefix $(SDBINDIR)/, $(mc2tasdcalibev_miscblist))

SDBINS += $(mc2tasdcalibev_miscbins) $(SDBINDIR)/sdmc_conv2tasdcalibev.run
.PHONY: mc2tasdcalibev
mc2tasdcalibev: $(mc2tasdcalibev_miscbins) $(SDBINDIR)/sdmc_conv2tasdcalibev.run
all:: mc2tasdcalibev

$(mc2tasdcalibev_miscbins): $(SDBINDIR)/%.run : $(MC2TASDCALIBEV)/src/%.o ; \
$(LD) $(LDFLAGS) $^ $(mc2tasdcalibev_libs) -o $@

mc2tasdcalibev_objlist = src/mc2tasdcalibev_class.o src/sdmc_conv2tasdcalibev.o
mc2tasdcalibev_objs = $(addprefix $(MC2TASDCALIBEV)/, $(mc2tasdcalibev_objlist))

$(SDBINDIR)/sdmc_conv2tasdcalibev.run: $(mc2tasdcalibev_objs) ; \
$(LD) $(LDFLAGS) $^ $(mc2tasdcalibev_libs) -o $@

clean:: ; \
rm -f $(MC2TASDCALIBEV)/src/*.o $(MC2TASDCALIBEV)/*~ $(MC2TASDCALIBEV)/src/*~ ; \

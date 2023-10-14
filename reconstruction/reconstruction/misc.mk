
misc_libs = -L$(SDLIBDIR) -lsden -lsduti $(ROOTLIBS_ALT) $(DSTLIBS)
misc_binlist = fdtoffset.run addrusdcal.run reduceDST.run
misc_binlist += printevtime.run sd4radar.run sdbsearch.run
misc_binfiles = $(addprefix $(SDBINDIR)/, $(misc_binlist))

SDBINS += $(misc_binfiles)

.PHONY: misc
misc: $(misc_binfiles)
all:: misc

$(misc_binfiles): $(SDBINDIR)/%.run : $(MISC)/%.o ; \
$(LD) $(LDFLAGS) -o $@ $^ $(misc_libs)

clean:: ; \
rm -f $(MISC)/*.o $(MISC)/*~


# creating the temporary directory if it doesn't exist
sdatm_tmp = $(SDATM)/tmp
$(if $(shell mkdir -p $(sdatm_tmp) && test -d $(sdatm_tmp) || echo "failed"),\
        $(error Failed to create the directory $(sdatm_tmp)),)


INCS += -I$(SDATM)
sdatm_libs = -L$(SDLIBDIR) -lsdfdrt -lsduti $(ROOTLIBS) $(DSTLIBS)
sdatm_cpplist = atmparfitter.cxx
sdatm_src = $(addprefix $(SDATM)/,$(sdatm_cpplist))
sdatm_src += $(sdatm_tmp)/atmparfitterDict.cxx
sdatm_objs = ${sdatm_src:.cxx=.o}
sdatm_h_list = atmparfitter.h atmparfitterLinkDef.h
sdatm_h = $(addprefix $(SDATM)/, $(sdatm_h_list))


sdatm_binlist   = sdatm_calib.run 
sdatm_binlist  += sdatm_corr.run
sdatm_binlist  += sdatm_sdascii.run
sdatm_binlist  += atmpar.run

sdatm_binfiles = $(addprefix $(SDBINDIR)/, $(sdatm_binlist))

SDSOLIBS   += $(SDLIBDIR)/libatmparfitter.so
SDSASOLIBS += $(SDLIBDIR)/libatmparfitter.so
SDBINS     += $(sdatm_binfiles)

.PHONY: atmparfitter
atmparfitter: $(SDLIBDIR)/libatmparfitter.so $(sdatm_binfiles)
all:: atmparfitter

$(SDLIBDIR)/libatmparfitter.so : $(sdatm_objs) ; \
$(LD) $(OPTOPT) -shared $^ $(sdatm_libs) -o $@ ; \
find $(SDATM) -name "*.pcm" -exec mv {} $(SDLIBDIR)/. \;

$(sdatm_tmp)/atmparfitterDict.cxx: $(SDATM)/atmparfitter.h $(SDATM)/atmparfitterLinkDef.h ; \
rootcint -f $@ $(ROOTCINTFLAGS) $(INCS) $(<F) $(filter %LinkDef.h, $^)

$(sdatm_binfiles): $(SDBINDIR)/%.run : $(SDATM)/%.o $(sdatm_objs); \
$(LD) $(LDFLAGS) -o $@ $^ $(sdatm_libs)

clean:: ; \
rm -f $(SDATM)/*.o $(SDATM)/*~ $(SDATM)/*Dict* $(SDATM)/*.pcm ; \
rm -rf $(sdatm_tmp)

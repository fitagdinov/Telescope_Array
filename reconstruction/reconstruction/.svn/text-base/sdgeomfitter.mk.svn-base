
INCS += -I$(SDGEOMFITTER)/inc

sdgeomfitter_cpplist = sdgeomfitter.cxx sdgeomfitterDict.cxx
sdgeomfitter_src = $(addprefix $(SDGEOMFITTER)/,$(sdgeomfitter_cpplist))
sdgeomfitter_objs = ${sdgeomfitter_src:.cxx=.o}
sdgeomfitter_h_list = sdgeomfitter.h sdgeomfitterLinkDef.h
sdgeomfitter_h = $(addprefix $(SDGEOMFITTER)/, $(sdgeomfitter_h_list))

SDSOLIBS += $(SDLIBDIR)/libsdgeomfitter.so
SDEDSO += $(SDLIBDIR)/libsdgeomfitter.so
.PHONY: sdgeomfitter
sdgeomfitter: $(SDLIBDIR)/libsdgeomfitter.so
all:: sdgeomfitter

$(SDLIBDIR)/libsdgeomfitter.so : $(sdgeomfitter_objs) ; \
$(CPP) $(OPTOPT) -shared $^ -o $@ ; \
find $(SDGEOMFITTER) -name "*.pcm" -exec mv {} $(SDLIBDIR)/. \;

$(SDGEOMFITTER)/sdgeomfitterDict.cxx: $(sdgeomfitter_h) ; \
rootcint -f $@ $(ROOTCINTFLAGS) $(INCS) $^

clean:: ; \
rm -f $(SDGEOMFITTER)/*.o $(SDGEOMFITTER)/*~ $(SDGEOMFITTER)/*Dict* $(SDGEOMFITTER)/*.pcm


sdfdrt_cxxlist = sdfdrt_class.cxx sdrt_class.cxx fdrt_class.cxx sdfdrtDict.cxx
sdfdrt_srcs = $(addprefix $(SDFDRT)/, $(sdfdrt_cxxlist))
sdfdrt_objs = ${sdfdrt_srcs:.cxx=.o}

SDLIBS += $(SDLIBDIR)/libsdfdrt.a

.PHONY: sdfdrt
sdfdrt: $(SDLIBDIR)/libsdfdrt.a
all:: sdfdrt

$(SDLIBDIR)/libsdfdrt.a: $(sdfdrt_objs) ; \
$(AR) rcs $@ $^; \
find $(SDFDRT) -name "*.pcm" -exec mv {} $(SDLIBDIR)/. \;

sdfdrt_header_list = sdfdrt_class.h sdrt_class.h fdrt_class.h LinkDef.h
sdfdrt_headers = $(addprefix $(SDFDRT)/, $(sdfdrt_header_list))

$(SDFDRT)/sdfdrtDict.cxx: $(sdfdrt_headers) ; \
rootcint -f $@ $(ROOTCINTFLAGS) $(INCS) $^

clean:: ; \
rm -f $(SDFDRT)/*.o $(SDFDRT)/*Dict.h $(SDFDRT)/*Dict.cxx $(SDFDRT)/*~ $(SDFDRT)/*.pcm


INCS += -I$(GLDFFITTER)/inc

gldffitter_cpplist = gldffitter.cxx gldffitterDict.cxx
gldffitter_src = $(addprefix $(GLDFFITTER)/,$(gldffitter_cpplist))
gldffitter_objs = ${gldffitter_src:.cxx=.o}
gldffitter_h_list = gldffitter.h gldffitterLinkDef.h
gldffitter_h = $(addprefix $(GLDFFITTER)/, $(gldffitter_h_list))

SDSOLIBS += $(SDLIBDIR)/libgldffitter.so
SDEDSO += $(SDLIBDIR)/libgldffitter.so
.PHONY: gldffitter
gldffitter: $(SDLIBDIR)/libgldffitter.so
all:: gldffitter

$(SDLIBDIR)/libgldffitter.so : $(gldffitter_objs) ; \
$(CPP) $(OPTOPT) -shared $^ -o $@ ; \
find $(GLDFFITTER) -name "*.pcm" -exec mv {} $(SDLIBDIR)/. \;

$(GLDFFITTER)/gldffitterDict.cxx: $(gldffitter_h) ; \
rootcint -f $@ $(ROOTCINTFLAGS) $(INCS) $^

clean:: ; \
rm -f $(GLDFFITTER)/*.o $(GLDFFITTER)/*~ $(GLDFFITTER)/*Dict* $(GLDFFITTER)/*.pcm

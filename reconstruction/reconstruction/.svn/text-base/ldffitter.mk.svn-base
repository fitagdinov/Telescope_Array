
INCS += -I$(LDFFITTER)/inc

ldffitter_cpplist = ldffitter.cxx ldffitterDict.cxx
ldffitter_src = $(addprefix $(LDFFITTER)/,$(ldffitter_cpplist))
ldffitter_objs = ${ldffitter_src:.cxx=.o}
ldffitter_h_list = ldffitter.h ldffitterLinkDef.h
ldffitter_h = $(addprefix $(LDFFITTER)/, $(ldffitter_h_list))

SDSOLIBS += $(SDLIBDIR)/libldffitter.so
SDEDSO += $(SDLIBDIR)/libldffitter.so
.PHONY: ldffitter
ldffitter: $(SDLIBDIR)/libldffitter.so
all:: ldffitter

$(SDLIBDIR)/libldffitter.so : $(ldffitter_objs) ; \
$(CPP) $(OPTOPT) -shared $^ -o $@ ; \
find $(LDFFITTER) -name "*.pcm" -exec mv {} $(SDLIBDIR)/. \;

$(LDFFITTER)/ldffitterDict.cxx: $(ldffitter_h) ; \
rootcint -f $@ $(ROOTCINTFLAGS) $(INCS) $^

clean:: ; \
rm -f $(LDFFITTER)/*.o $(LDFFITTER)/*~ $(LDFFITTER)/*Dict* $(LDFFITTER)/*.pcm


INCS += -I$(RUSDPASS0)/inc
rusdpass0_libs = -L$(SDLIBDIR) -lsduti $(ROOTLIBS_ALT) $(DSTLIBS) 
rusdpass0_cpplist = rusdpass0.cpp parsingmanager.cpp tower_parser.cpp rusdpass0io.cpp sdindexing.cpp
rusdpass0_srcs = $(addprefix $(RUSDPASS0)/src/, $(rusdpass0_cpplist))
rusdpass0_objs = ${rusdpass0_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/rusdpass0.run
rusdpass0: $(SDBINDIR)/rusdpass0.run
all:: rusdpass0

$(SDBINDIR)/rusdpass0.run: $(rusdpass0_objs) ; \
$(LD) $(LDFLAGS) $^ $(rusdpass0_libs) -o $@ ; \

clean:: ; \
rm -f $(RUSDPASS0)/*~ $(rusdpass0_objs) $(RUSDPASS0)/src/*~ ; \

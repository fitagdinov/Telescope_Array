
sdenergy_objlist = rusdenergy.o
sdenergy_obj = $(addprefix $(SDENERGY)/, $(sdenergy_objlist))

SDLIBS += $(SDLIBDIR)/libsden.a
.PHONY: sdenergy
sdenergy: $(SDLIBDIR)/libsden.a
all:: sdenergy

$(SDLIBDIR)/libsden.a: $(sdenergy_obj) ;\
$(AR) rcs $@ $^; \

clean:: ;\
rm -f $(SDENERGY)/*.o $(SDENERGY)/*~

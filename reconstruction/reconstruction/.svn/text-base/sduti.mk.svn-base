

sduti_objlist = sdio.o sdgen.o tacoortrans.o sdxyzclf_class.o sdparamborder.o 
sduti_objlist += icrr2ru.o sdstdz76.o sdckatm.o sdatmos.o tafd10info_time.o 
sduti_objlist += tafd10info_tubes.o tafd10info_banks.o sddstio.o
sduti_objlist += sdmc_bsd_bitf.o sdmc_tadate.o sdgdas.o sdinterp.o

sduti_obj = $(addprefix $(SDUTI)/, $(sduti_objlist))

SDLIBS += $(SDLIBDIR)/libsduti.a
.PHONY: sduti
sduti: $(SDLIBDIR)/libsduti.a
all:: sduti

$(SDLIBDIR)/libsduti.a: $(sduti_obj) ;\
$(AR) rcs $@ $^; \

clean:: ;\
rm -f $(SDUTI)/*.o $(SDUTI)/*~

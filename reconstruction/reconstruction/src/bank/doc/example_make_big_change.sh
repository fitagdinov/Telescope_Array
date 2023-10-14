#!/usr/bin/env bash



PROTO_BANKS=()
DERIVED_BANKS=()
DERIVED_BANK_PROTO=()
derlist="br lr md tl stereo hy brhy lrhy"
for h in inc/*fd*_dst.h; do
    h_is_a_base=0
    derived_base=$(echo $h | sed 's/.*\/\([a-z,A-Z,_,0-9]*\)_dst\.h/\1/')
    for fd in $derlist; do
	f=$(echo $h | sed "s/fd/${fd}/")
	if [ -f $f ]; then
	    h_is_a_base=1
	    derived_name=$(echo $f | sed 's/.*\/\([a-z,A-Z,_,0-9]*\)_dst\.h/\1/')
	    DERIVED_BANKS+=("${derived_name}")
	    DERIVED_BANK_PROTO+=("${derived_base}")
	fi
    done
    if [ $h_is_a_base -gt 0 ]; then
	PROTO_BANKS+=("${derived_base}")
    fi
done

PROTO_BANKS+=("hyp1")
DERIVED_BANKS+=("brhyp1")
DERIVED_BANK_PROTO+=("hyp1")
DERIVED_BANKS+=("lrhyp1")
DERIVED_BANK_PROTO+=("hyp1")
DERIVED_BANKS+=("mdhyp1")
DERIVED_BANK_PROTO+=("hyp1")


DSTBANKS=$(ls src/bank/lib/*_dst.c | sed 's/.*\/\([a-z,A-Z,_,0-9]*\)_dst\.c/\1/')


n_warnings=0
n_skipped=0


for bank in ${DSTBANKS}; do
    
    header=inc/${bank}_dst.h
    source=src/bank/lib/${bank}_dst.c
    proto_bank=""
    is_proto_bank=0
    is_derived_bank=0

    # SKIP START ADN STOP BANKS
    test "${bank}" == "start" && echo "SKIPPING ${bank}" && n_skipped=$((n_skipped+1)) && continue
    test "${bank}" == "stop" && echo "SKIPPING ${bank}" && n_skipped=$((n_skipped+1)) && continue

    # SKIP IF ALREADY DONE
    n=$(grep '_bank_buffer_size)' $header | wc -l)
    test $n -gt 0 && echo "SKIPPING ${bank}" && n_skipped=$((n_skipped+1)) && continue
    n=$(grep 'int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*' -B3 $header | grep '__cplusplus' | wc -l)
    test $n -gt 0 && echo "SKIPPING ${bank}" && n_skipped=$((n_skipped+1)) && continue
    n=$(grep '_bank_buffer_size)' $source | wc -l)
    test $n -gt 0 && echo "SKIPPING ${bank}" && n_skipped=$((n_skipped+1)) && continue

    # NORMALIZE SOME SOURCE FILES: SOME VARIABLES SHOULD BE LOWER CASE
    sed -i 's/\([0-9,A-Z,_]*\)_BLEN/\L\1_BLEN/g' $source
    sed -i 's/\([0-9,A-Z,_]*\)_MAXLEN/\L\1_MAXLEN/g' $source	
    sed -i 's/\([0-9,A-Z,_]*\)_BANK\([^VIS_]\)/\L\1_BANK\2/g' $source
    sed -i 's/\([0-9,A-Z,_]*\)_BLOCK\([^VIS_]\)/\L\1_BANK\2/g' $source


    # check if a prototype bank
    for abank in "${PROTO_BANKS[@]}"; do
	if [ "${bank}" == "${abank}" ]; then
	    is_proto_bank=1
	    proto_bank=${abank}
	    break
	fi
    done
    # check if derived
    for i in "${!DERIVED_BANKS[@]}"; do
	if [ "${bank}" == "${DERIVED_BANKS[${i}]}" ]; then
	    is_derived_bank=1
	    proto_bank=${DERIVED_BANK_PROTO[${i}]}
	    break
	fi
    done

    
    # CHECK HEADER
    n=$(grep 'int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*' $header | wc -l) 
    if [ $n -eq 0 ]; then
	n=$(grep 'int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_bank_to_common_.*' $header | wc -l)
	test $n -eq 0 && echo "WARNING: HEADER NOT MATCHING: ${header}" && n_warnings=$((n_warnings+1)) && continue
    fi
    n=$(grep 'int[e,g,e,r,4]*\s*\([a-z,0-9,A-Z,_]*\)_common_to_dumpf_.*' $header | wc -l)
    test $n -eq 0 && echo "WARNING: HEADER NOT MATCHING: ${header}" && n_warnings=$((n_warnings+1)) && continue

    # CHECK SOURCE
    n=$(grep -i 'static\s*[a-z,0-9]*\s*\*\s*\([a-z,0-9,A-Z,_]*\)_bank\s*=\s*NULL.*;' $source | wc -l)
    test $n -eq 0 && echo "WARNING: SOURCE NOT MATCHING: ${source}" && n_warnings=$((n_warnings+1)) && continue
    
    # ADDITIONAL CHECKS FOR PROTOTYPE HEADER AND SOURCE
    if [ ${is_proto_bank} -eq 1 ]; then
	n=$(grep -i "static\s*int[e,g,e,r,4]*\s*${bank}_blen\s*;" $source | wc -l)
	test $n -eq 0 && echo "WARNING: PROTO SOURCE NOT MATCHING: ${source}" && n_warnings=$((n_warnings+1)) && continue

	n=$(grep -i "extern\s*${bank}_dst_common\s*${bank}_\s*;" $header | wc -l)
	test $n -eq 0 && echo "WARNING: PROTO HEADER NOT MATCHING: ${header}" && n_warnings=$((n_warnings+1)) && continue
    fi
    
    # PROTOTYPE BANK
    if [ ${is_proto_bank} -eq 1 ]; then
	# header
	n=$(grep 'int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*' $header | wc -l)
	if [ $n -gt 0 ]; then
	    sed -i 's/int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*/#ifdef __cplusplus\nextern "C" {\n#endif\n&/' $header
	else
	    sed -i 's/int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_bank_to_common_.*/#ifdef __cplusplus\nextern "C" {\n#endif\n&/' $header
	fi
	sed -i 's/int[e,g,e,r,4]*\s*\([a-z,0-9,A-Z,_]*\)_common_to_dumpf_.*/&\n\1_ADDITIONAL_CODE_GOES_HERE_\n#ifdef __cplusplus\n} \/\/end extern "C"\n#endif\n/' $header
	sed -i 's/\(.*\)_ADDITIONAL_CODE_GOES_HERE_/\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size);/' $header
	sed -i "s/extern ${bank}_dst_common\s*${bank}_\s*;/&\nextern integer4 ${bank}_blen; \/\* needs to be accessed by the c files of the derived banks \*\/ /" $header
	
	# source
	sed -i 's/static\s*[a-z,0-9]*\s*\*\s*\([a-z,0-9,A-Z,_]*\)_bank\s*=\s*NULL.*;/&\n\n\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size)\n{\n  (*\1_bank_buffer_size) = \1_blen;\n  return \1_bank;\n}\n\n/' $source
	sed -i 's/static\s*[a-z,0-9]*\s*\*\s*\([a-z,0-9,A-Z,_]*\)_BANK\s*=\s*NULL.*;/&\n\n\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size)\n{\n  (*\1_bank_buffer_size) = \1_blen;\n  return \1_bank;\n}\n\n/' $source
	sed -i "s/static\s*int[e,g,e,r,4]*\s*${bank}_blen\s*;/integer4 ${bank}_blen = 0; \/\* not static because it needs to be accessed by the c files of the derived banks \*\//" $source
	
	continue
    fi

    # DERIVED BANK
    if [ ${is_derived_bank} -eq 1 ]; then
	
	# header
	n=$(grep 'int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*' $header | wc -l)
	if [ $n -gt 0 ]; then
	    sed -i 's/int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*/#ifdef __cplusplus\nextern "C" {\n#endif\n&/' $header
	else
	    sed -i 's/int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_bank_to_common_.*/#ifdef __cplusplus\nextern "C" {\n#endif\n&/' $header
	fi
	sed -i 's/int[e,g,e,r,4]*\s*\([a-z,0-9,A-Z,_]*\)_common_to_dumpf_.*/&\n\1_ADDITIONAL_CODE_GOES_HERE_\n#ifdef __cplusplus\n} \/\/end extern "C"\n#endif\n/' $header
	sed -i 's/\(.*\)_ADDITIONAL_CODE_GOES_HERE_/\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size);/' $header
	
	# source
	sed -i "s/static\s*[a-z,0-9]*\s*\*\s*\([a-z,0-9,A-Z,_]*\)_bank\s*=\s*NULL.*;/&\n\n\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size)\n{\n  (*\1_bank_buffer_size) = ${proto_bank}_blen;\n  return \1_bank;\n}\n\n/" $source
	sed -i "s/static\s*[a-z,0-9]*\s*\*\s*\([a-z,0-9,A-Z,_]*\)_BANK\s*=\s*NULL.*;/&\n\n\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size)\n{\n  (*\1_bank_buffer_size) = ${proto_bank}_blen;\n  return \1_bank;\n}\n\n/" $source
	
	continue
    fi


    
    # ALL OTHER (NORMAL) BANKS


   # header
    n=$(grep 'int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*' $header | wc -l)
    if [ $n -gt 0 ]; then
	sed -i 's/int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_common_to_bank_.*/#ifdef __cplusplus\nextern "C" {\n#endif\n&/' $header
    else
	sed -i 's/int[e,g,e,r,4]*\s*[a-z,0-9,A-Z,_]*_bank_to_common_.*/#ifdef __cplusplus\nextern "C" {\n#endif\n&/' $header
    fi
    sed -i 's/int[e,g,e,r,4]*\s*\([a-z,0-9,A-Z,_]*\)_common_to_dumpf_.*/&\n\1_ADDITIONAL_CODE_GOES_HERE_\n#ifdef __cplusplus\n} \/\/end extern "C"\n#endif\n/' $header
    sed -i 's/\(.*\)_ADDITIONAL_CODE_GOES_HERE_/\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size);/' $header
    
    # source
    sed -i 's/static\s*[a-z,0-9]*\s*\*\s*\([a-z,0-9,A-Z,_]*\)_bank\s*=\s*NULL.*;/&\n\n\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size)\n{\n  (*\1_bank_buffer_size) = \1_blen;\n  return \1_bank;\n}\n\n/' $source
    sed -i 's/static\s*[a-z,0-9]*\s*\*\s*\([a-z,0-9,A-Z,_]*\)_BANK\s*=\s*NULL.*;/&\n\n\/* get (packed) buffer pointer and size *\/\ninteger1* \1_bank_buffer_ (integer4* \1_bank_buffer_size)\n{\n  (*\1_bank_buffer_size) = \1_blen;\n  return \1_bank;\n}\n\n/' $source
    sed -i "s/static\s*int[e,g,e,r,4]*\s*${bank}_blen\s*;/static integer4 ${bank}_blen = 0;/" $source
 
    
done




echo "SKIPPED: ${n_skipped}"
echo "WARNINGS: ${n_warnings}"

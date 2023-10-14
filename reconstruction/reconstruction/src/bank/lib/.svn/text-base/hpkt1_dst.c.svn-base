/*
 * 
 * hpkt1_dst.c
 * 
 * $Source: /hires_soft/uvm2k/bank/hpkt1_dst.c,v $
 * $Log: hpkt1_dst.c,v $
 * Revision 1.8  2000/09/13 23:19:24  jeremy
 * Added HR_TYPE_CUTEVENT packet type
 *
 * Revision 1.7  1997/05/30 21:37:21  jui
 * fixed volts packet reading and dumping
 *
 * Revision 1.6  1997/05/09  23:39:34  vtodd
 * changed event structure to a packed list, eliminated duplicate
 * snapshot structure so that only event is used
 *
 * Revision 1.5  1997/05/09  17:01:42  jui
 * make dump formats for event, time, and notice packets
 *
 * Revision 1.4  1997/04/28  16:32:00  jui
 * made some corrections to the "dump" section for event packets
 *
 * Revision 1.3  1997/04/26  01:24:16  jui
 * broke up hpkt1_bank_to_packet_ into three pieces (essentially):
 * (1) hpkt1_unpack_bank_hdr_ (unpacks bank header)
 * (2) hpkt1_unpack_packet_hdr_ (unpacks packet header)
 * (3) hpkt1_unpack_packet_body_ (unpacks bbody of packet)
 * int the new scheme, hpkt1_unpack_packet_body_ assumes both
 * hpkt1_unpack_packet_hdr_ and hpkt1_unpack_packet_body_ are called
 * already. NOte it is not necessary really to call hpkt1_unpack_packet_body_
 * unless you intend to write out the packet banks again...
 *
 * Revision 1.2  1997/04/24  18:14:43  jui
 * broke up hpkt1_bank_to_common_ into two pieces:
 * (1) hpkt1_bank_to_packet_ (strips bank headers from packet)
 * (2) hpkt1_packet_to_common_ (unpacks packets into common block/structure)
 * hpkt1_bank_to_common_ simply calls (1) and (2) above in turn.
 *
 * Revision 1.1  1997/04/16  22:36:26  vtodd
 * Initial revision
 *
 * 
 */


#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "hpkt1_dst.h"  


/********************************************/
/* Allocate space for hpkt1 bank structures */

hpkt1_dst_raw        hpkt1_raw_;
hpkt1_dst_event      hpkt1_event_;
hpkt1_dst_time       hpkt1_time_;
hpkt1_dst_minute     hpkt1_minute_;
hpkt1_dst_threshold  hpkt1_threshold_;
hpkt1_dst_countrate  hpkt1_countrate_;
hpkt1_dst_notice     hpkt1_notice_;
hpkt1_dst_remote     hpkt1_remote_;
hpkt1_dst_calib      hpkt1_calib_;
hpkt1_dst_mstat      hpkt1_mstat_;
hpkt1_dst_volts      hpkt1_volts_;
hpkt1_dst_boardid    hpkt1_boardid_;

static integer4 hpkt1_maxlen = 2*sizeof(integer4)+HPKT1_MAX_PKT_LEN;
  static integer4 hpkt1_blen = 0;
  static integer1 *hpkt1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hpkt1_bank_buffer_ (integer4* hpkt1_bank_buffer_size)
{
  (*hpkt1_bank_buffer_size) = hpkt1_blen;
  return hpkt1_bank;
}


  
  
  static void hpkt1_bank_init(void)
{
  /*********************************************************************/
  /* Allocate space for bank id, bank version, and largest packet type */
  
  hpkt1_bank = (integer1 *)calloc(2*sizeof(integer4)+HPKT1_MAX_PKT_LEN,
                                  sizeof(integer1));
  if (hpkt1_bank==NULL)
    {
      fprintf (stderr,"hpkt1_bank_init: failed to allocate memory for bank.Abort.\n");
      exit(0);
    }
}


integer4 hpkt1_common_to_bank_(void)
{
  static integer4 id = HPKT1_BANKID, ver = HPKT1_BANKVERSION;
  integer4 rcode, nobj;
  
  if (hpkt1_bank == NULL) hpkt1_bank_init();
  
  /***************************************************************************/
  /* Initialize hpkt1_maxlen for bank id and version, packet header and size */
  
  hpkt1_maxlen = 2*sizeof(integer4)+4*sizeof(integer2)+hpkt1_raw_.pktHdr_size;
  
  /**************************************************************/
  /* Initialize hpkt1_blen, and pack the id and version to bank */
  
  rcode = dst_initbank_(&id, &ver, &hpkt1_blen, &hpkt1_maxlen, hpkt1_bank);
  
  /**************************/  
  /* Pack the packet header */
  
  if ( (rcode = dst_packi2_(&hpkt1_raw_.pktHdr_type, (nobj=1, &nobj), hpkt1_bank,
			    &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi2_(&hpkt1_raw_.pktHdr_crate, (nobj=1, &nobj),hpkt1_bank,
			    &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi2_(&hpkt1_raw_.pktHdr_id, (nobj=1, &nobj), hpkt1_bank,
			    &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi2_(&hpkt1_raw_.pktHdr_size, (nobj=1, &nobj), hpkt1_bank,
			    &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  
  /***********************/
  /* Pack the raw packet */
  
  if ( (rcode = dst_packi1_(&hpkt1_raw_.raw_pkt[0],
			    (nobj=hpkt1_raw_.pktHdr_size, &nobj), hpkt1_bank, 
			    &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  return rcode ;
}


integer4 hpkt1_bank_to_dst_ (integer4 *unit)
{
  return dst_write_bank_(unit, &hpkt1_blen, hpkt1_bank);
}


integer4 hpkt1_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = hpkt1_common_to_bank_()) )
    {
      fprintf(stderr, "hpkt1_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
    if ( (rcode = hpkt1_bank_to_dst_(unit) ))
    {
      fprintf(stderr, "hpkt1_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
  return 0;
}


integer4 hpkt1_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer1 *packet;

  if ( (rcode = hpkt1_unpack_bank_hdr_(bank)) )
     return rcode;

  /* pull out the packet */
  packet = bank + 2*sizeof(integer4);

  if ( (rcode = hpkt1_unpack_packet_hdr_(packet)) )
     return rcode;

  if ( (rcode = hpkt1_unpack_packet_body_(packet)) )
     return rcode;

  if ( (rcode = hpkt1_packet_to_common_(packet)) )
     return rcode;

  return 0;
}  
  
integer4 hpkt1_unpack_bank_hdr_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  hpkt1_blen = 0;
  hpkt1_maxlen = 2*sizeof(integer4)+4*sizeof(integer2)+HPKT1_MAX_PKT_LEN;
  
  if ( (rcode = dst_unpacki4_(&hpkt1_raw_.bank_id, (nobj=1, &nobj), bank,
			      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(&hpkt1_raw_.bank_ver, (nobj=1, &nobj), bank,
			      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  return rcode;
}

integer4 hpkt1_unpack_packet_hdr_(integer1 *packet)
{
  integer4 rcode = 0;
  integer4 nobj;
  
  hpkt1_blen = 0;
  hpkt1_maxlen = 4*sizeof(integer2)+HPKT1_MAX_PKT_LEN;

  /**************************************************************/ 
  /* Extracting the packet header information;                  */
  
  if ( (rcode = dst_unpacki2_(&hpkt1_raw_.pktHdr_type,  (nobj=1, &nobj), packet,
			      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2_(&hpkt1_raw_.pktHdr_crate,  (nobj=1, &nobj), packet,
			      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2_(&hpkt1_raw_.pktHdr_id,  (nobj=1, &nobj), packet,
			      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2_(&hpkt1_raw_.pktHdr_size,  (nobj=1, &nobj), packet,
			      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
  return rcode;
}

integer4 hpkt1_unpack_packet_body_(integer1 *packet)
{
  integer4 rcode = 0;
  integer4 nobj;

  hpkt1_blen = 4*sizeof(integer2);
  hpkt1_maxlen = hpkt1_raw_.pktHdr_size+4*sizeof(integer2);
  
  /************************************************************/ 
  /* Unpack bank into the raw packet structure extracting the */
  /* bank id and version and the packet header information    */
  
  if ( (rcode = dst_unpacki1_(&hpkt1_raw_.raw_pkt[0],
			      (nobj=hpkt1_raw_.pktHdr_size, &nobj), packet,
			      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;

  return rcode;
}  
  

integer4 hpkt1_packet_to_common_(integer1 *packet)
{
  integer4 rcode = 0, i, nobj;
  integer2 dum2;
  unsigned char tube_num, dum1;
  
  hpkt1_blen = 4*sizeof(integer2);
  hpkt1_maxlen = hpkt1_raw_.pktHdr_size+4*sizeof(integer2);

  /******************************************************************/
  /* Unpack the various packet structures according to packet type  */
  /* setting hpkt1_blen beyond bank id and version                  */
  
  switch(hpkt1_raw_.pktHdr_type) 
    {
    case HR_TYPE_EVENT:
    case HR_TYPE_SNAPSHOT:
      if ( (rcode = dst_unpacki4_(&hpkt1_event_.event, (nobj=1, &nobj), packet, 
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.version, (nobj=1, &nobj), 
				      packet,&hpkt1_blen, 
				      &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.minute, (nobj=1, &nobj), 
				      packet, &hpkt1_blen, 
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_event_.minute &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.msec, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_event_.msec &= 0xFFFF;

      /************************************************************/
      /* Unpack the number of tubes (ntubes) in the event and     */
      /* proceed to unpack the subsequent (ntube) data structures */
      
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.ntubes, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;

      for(i=0;i<hpkt1_event_.ntubes;i++) {
        if ( (rcode = dst_unpacki1_((integer1 *)&tube_num, (nobj=1, &nobj),
				    packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
        hpkt1_event_.tube_num[i] = (integer4)tube_num;
        
        if ( (rcode = dst_unpacki1_((integer1 *)&dum1, (nobj=1, &nobj),
				    packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
        dum2 = (integer2) dum1 * 0x0010;
        if ( (rcode = dst_unpacki1_((integer1 *)&dum1, (nobj=1, &nobj),
				    packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
        hpkt1_event_.thA[i] = dum2 + ( (integer2)dum1 / 0x0010 );
        
        dum2 = ( (integer2)dum1 & 0x000f ) * 0x0100;
        if ( (rcode = dst_unpacki1_((integer1 *)&dum1, (nobj=1, &nobj),
				    packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
        hpkt1_event_.thB[i] = dum2 +( (integer2)dum1 );
        
        if ( (rcode = dst_unpacki2_(&hpkt1_event_.qdcA[i], 
				    (nobj=1, &nobj), packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_event_.qdcB[i],
				    (nobj=1, &nobj), packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_event_.tdc[i], (nobj=1, &nobj),
				    packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
      }
      break;
      
    case HR_TYPE_CUTEVENT:
      if ( (rcode = dst_unpacki4_(&hpkt1_event_.event, (nobj=1, &nobj), packet, 
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.version, (nobj=1, &nobj), 
				      packet,&hpkt1_blen, 
				      &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.minute, (nobj=1, &nobj), 
				      packet, &hpkt1_blen, 
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_event_.minute &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.msec, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_event_.msec &= 0xFFFF;

      if ( (rcode = dst_unpacki2asi4_(&hpkt1_event_.ntubes, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      break;

    case HR_TYPE_TIME:
      if ( (rcode = dst_unpacki2_(&hpkt1_time_.year, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_time_.day, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_time_.sec, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_time_.freq, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_time_.mark_error, (nobj=1, &nobj),
				  packet, &hpkt1_blen, 
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_time_.minute_offset, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_time_.error_flags, (nobj=1, &nobj), 
				  packet, &hpkt1_blen, 
				  &hpkt1_maxlen)) ) return rcode;
      
      /******************************************************/
      /* Unpack the number of mirror events (n) and proceed */
      /* to unpack the n subsequent time data structures    */
      
      if ( (rcode = dst_unpacki2_(&hpkt1_time_.events, (nobj=1, &nobj), packet, 
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      
      for(i=0;i<hpkt1_time_.events;i++) {
        if ( (rcode = dst_unpacki2_(&hpkt1_time_.mirror[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_time_.msec[i], (nobj=1, &nobj), packet,
				    &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki4_(&hpkt1_time_.nsec[i], (nobj=1, &nobj), packet,
				    &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      }
      break;
      
      
    case HR_TYPE_MINUTE:
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_minute_.minute, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_minute_.minute &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_minute_.trigs, (nobj=1, &nobj), 
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_minute_.trigs &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_minute_.msec, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_minute_.msec &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_minute_.dead, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_minute_.dead &= 0xFFFF;
      break;
      
      
    case HR_TYPE_COUNTRATE:
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_countrate_.min, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_blen += 2;    /* skip 2 bytes of padding */
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_countrate_.cntRateA[0],
				      (nobj=HR_UNIV_MIRTUBE, &nobj), packet, 
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      for (i = 0; i < HR_UNIV_MIRTUBE; ++i)
        hpkt1_countrate_.cntRateA[i] &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_countrate_.cntRateB[0],
				      (nobj=HR_UNIV_MIRTUBE, &nobj), packet, 
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      for (i = 0; i < HR_UNIV_MIRTUBE; ++i)
        hpkt1_countrate_.cntRateB[i] &= 0xFFFF;
      break;
      
      
    case HR_TYPE_THRESHOLD:
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_threshold_.min, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_blen += 2;    /* skip 2 bytes of padding */
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_threshold_.thA[0],
				      (nobj=HR_UNIV_MIRTUBE, &nobj), packet, 
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      for (i = 0; i < HR_UNIV_MIRTUBE; ++i)
        hpkt1_threshold_.thA[i] &= 0x0FFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_threshold_.thB[0],
				      (nobj=HR_UNIV_MIRTUBE, &nobj), packet, 
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      for (i = 0; i < HR_UNIV_MIRTUBE; ++i)
        hpkt1_threshold_.thB[i] &= 0x0FFF;
      break;
      
      
    case HR_TYPE_NOTICE:
      if ( (rcode = dst_unpacki4_(&hpkt1_notice_.type, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_notice_.year, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_notice_.day, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_notice_.hour, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_notice_.min, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_notice_.sec, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_notice_.msec, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki1_(&hpkt1_notice_.text[0],
				  (nobj=hpkt1_raw_.pktHdr_size-16, &nobj),
				  packet, &hpkt1_blen, 
				  &hpkt1_maxlen)) ) return rcode;
      break;


    case HR_TYPE_REMOTE:
      if ( (rcode = dst_unpacki1_(&hpkt1_remote_.tag[0], (nobj=8, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki1_(&hpkt1_remote_.text[0],
				  (nobj=hpkt1_raw_.pktHdr_size-8, &nobj),
				  packet, &hpkt1_blen, 
				  &hpkt1_maxlen)) ) return rcode;
      break;
      
      
    case HR_TYPE_CALIB:
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_calib_.type, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_calib_.type &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_calib_.count, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_calib_.count &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_calib_.ampl, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_calib_.ampl &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_calib_.width, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_calib_.width &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_calib_.period, (nobj=1, &nobj), 
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_calib_.period &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_calib_.delay, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_calib_.delay &= 0xFFFF;
      if ( (rcode = dst_unpacki2_(&hpkt1_calib_.mean[0],
				  (nobj=HR_UNIV_MIRTUBE, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_calib_.sdev[0],
				  (nobj=HR_UNIV_MIRTUBE, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      break;

      
    case HR_TYPE_MSTAT:
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.sent, (nobj=1, &nobj), packet, 
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.resent, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.rcvd, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.rercvd, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.errs, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.warns, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.lost, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.halts, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki4_(&hpkt1_mstat_.maxMsgs, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      break;
      
      
    case HR_TYPE_VOLTS:
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_volts_.minute, (nobj=1, &nobj),
				      packet, &hpkt1_blen, 
				      &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.obVer, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.hvChnls, (nobj=1, &nobj), packet,
				  &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;

      /*****************************************************/
      /* Unpack the 16 ommatidial board readout structures */
      
      for(i=0;i<16;i++) {
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_p12v[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_p05v[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_n12v[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_n05v[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_tdcRef[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_temp[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_thRef[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.ob_gnd[i], (nobj=1, &nobj), 
				    packet, &hpkt1_blen, 
				    &hpkt1_maxlen)) ) return rcode;
      }       
      
      /**********************************************/
      /* Unpack the garbage board readout structure */
      
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_temp, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_p12v, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_n12v, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_p05v, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_s05v, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_lemo1, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_anlIn, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_clsVolts, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_clsTemp, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_mirX, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_mirY, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_clsX, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_clsY, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_ns, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_hvSup, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.garb_hvChnl, (nobj=1, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      
      /**********************************************************************/
      /* Unpack cluster mux ADC channels and HV readback from garbage board */
      
      if ( (rcode = dst_unpacki2_(&hpkt1_volts_.cluster[0], (nobj=16, &nobj),
				  packet, &hpkt1_blen,
				  &hpkt1_maxlen)) ) return rcode;
      for(i=0;i<hpkt1_volts_.hvChnls;i++) {
        if ( (rcode = dst_unpacki2_(&hpkt1_volts_.hv[i], (nobj=1, &nobj),
				    packet, &hpkt1_blen,
				    &hpkt1_maxlen)) ) return rcode;
      }
      break;

      
    case HR_TYPE_BOARDID:
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_boardid_.version, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_boardid_.version &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_boardid_.cpu, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_boardid_.cpu &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_boardid_.spare, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_boardid_.spare &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_boardid_.ppg, (nobj=1, &nobj), packet,
				      &hpkt1_blen, &hpkt1_maxlen)) ) return rcode;
      hpkt1_boardid_.ppg &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_boardid_.trig, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_boardid_.trig &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_boardid_.ob[0], (nobj=16, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      for (i = 0; i < 16; ++i) hpkt1_boardid_.ob[i] &= 0xFFFF;
      if ( (rcode = dst_unpacki2asi4_(&hpkt1_boardid_.garb, (nobj=1, &nobj),
				      packet, &hpkt1_blen,
				      &hpkt1_maxlen)) ) return rcode;
      hpkt1_boardid_.garb &= 0xFFFF;
      break;
    }
  return rcode ;
}

integer4 hpkt1_common_to_dump_(integer4 *long_output)
{
  return hpkt1_common_to_dumpf_(stdout,long_output);
}


integer4 hpkt1_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i, cnt, hr, min, sec;
  integer1 *ptr_text_buf;  
  
  switch(hpkt1_raw_.pktHdr_type) 
    {
      
    case HR_TYPE_EVENT:
    case HR_TYPE_SNAPSHOT:
      if(hpkt1_raw_.pktHdr_type == HR_TYPE_EVENT){
	fprintf (fp, "HPKT1 crate %2d EVENT %6d Rev %1d Time +%3.3d:%2.2d.%3.3d Tubes %3d\n",
		 hpkt1_raw_.pktHdr_crate, hpkt1_event_.event,
		 hpkt1_event_.version, hpkt1_event_.minute,
		 hpkt1_event_.msec/1000,hpkt1_event_.msec%1000,
		 hpkt1_event_.ntubes);
      } else {
	fprintf (fp, "HPKT1 crate %2d SNAPSHOT %6d Rev %1d Time +%3.3d:%2.2d.%3.3d Tubes %3d\n",
		 hpkt1_raw_.pktHdr_crate, hpkt1_event_.event,
		 hpkt1_event_.version, hpkt1_event_.minute,
		 hpkt1_event_.msec/1000,hpkt1_event_.msec%1000,
		 hpkt1_event_.ntubes);
      }
      if(*long_output == 1) {
        for(i=0;i<hpkt1_event_.ntubes;i++) {
	  fprintf (fp,
		   "tube %3d qdcA %5d thA %4d qdcB %5d thB %4d tdc %5d\n",
		   hpkt1_event_.tube_num[i], hpkt1_event_.qdcA[i],
		   hpkt1_event_.thA[i], hpkt1_event_.qdcB[i],
		   hpkt1_event_.thB[i], hpkt1_event_.tdc[i]);
        }
      }
      break;

    case HR_TYPE_CUTEVENT:
      fprintf (fp, "HPKT1 crate %2d CUTEVENT %6d Rev %1d Time +%3.3d:%2.2d.%3.3d Tubes %3d\n",
	       hpkt1_raw_.pktHdr_crate, hpkt1_event_.event,
	       hpkt1_event_.version, hpkt1_event_.minute,
	       hpkt1_event_.msec/1000,hpkt1_event_.msec%1000,
	       hpkt1_event_.ntubes);
      break;

    case HR_TYPE_TIME:
      hr  = hpkt1_time_.sec/3600;
      min = (hpkt1_time_.sec%3600)/60;
      sec = hpkt1_time_.sec%60;
      fprintf (fp, "HPKT1 crate %2d TIME %04d/%03d %02d:%02d:%02d events %3d\n",
               hpkt1_raw_.pktHdr_crate, hpkt1_time_.year, hpkt1_time_.day,
               hr, min, sec, hpkt1_time_.events);
      if(*long_output == 1) {
        fprintf (fp, "freq %8d mark_err %3d min_off %2d err_flags %04X\n",
                 hpkt1_time_.freq, hpkt1_time_.mark_error, 
                 hpkt1_time_.minute_offset, hpkt1_time_.error_flags);
        for(cnt=0;cnt<hpkt1_time_.events;cnt++) {
          fprintf (fp,"mirror %2d msec %5d nsec %6d\n",
                   hpkt1_time_.mirror[cnt], hpkt1_time_.msec[cnt],
                   hpkt1_time_.nsec[cnt]);
        }
      }
      break;

      
    case HR_TYPE_MINUTE:
      fprintf (fp,
               "HPKT1 crate %2d MINUTE %3d trigs %6d msec %5u dead %5u\n",
               hpkt1_raw_.pktHdr_crate, hpkt1_minute_.minute,
               hpkt1_minute_.trigs, hpkt1_minute_.msec,
               hpkt1_minute_.dead);
      break;
      
      
    case HR_TYPE_COUNTRATE:
      fprintf (fp, "HPKT1 crate %2d COUNTRATE min %3d\n",
               hpkt1_raw_.pktHdr_crate, hpkt1_countrate_.min);
      if(*long_output == 1) {
        fprintf (fp,"countRateA:\n");
        for(cnt=0;cnt<HR_UNIV_MIRTUBE;cnt++) {
          if(cnt%16 == 0) fprintf(fp, "\n");
          fprintf (fp,"%5d ", hpkt1_countrate_.cntRateA[cnt]);
        }
        fprintf (fp,"\ncountRateB:\n");
        for(cnt=0;cnt<HR_UNIV_MIRTUBE;cnt++) {
          if(cnt%16 == 0) fprintf(fp, "\n");
          fprintf (fp,"%5d ", hpkt1_countrate_.cntRateB[cnt]);
        }
        fprintf (fp,"\n");
      } 
      break;


    case HR_TYPE_THRESHOLD:
      fprintf (fp, "HPKT1 crate %2d THRESHOLD min %3d\n",
               hpkt1_raw_.pktHdr_crate, hpkt1_threshold_.min);
      if(*long_output == 1) {
        fprintf (fp,"thresholdA:\n");
        for(cnt=0;cnt<HR_UNIV_MIRTUBE;cnt++) {
          if(cnt%16 == 0) fprintf(fp, "\n");
          fprintf (fp,"%5d ", hpkt1_threshold_.thA[cnt]);
        }
        fprintf (fp,"\nthreholdB:\n");
        for(cnt=0;cnt<HR_UNIV_MIRTUBE;cnt++) {
          if(cnt%16 == 0) fprintf(fp, "\n");
          fprintf (fp,"%5d ", hpkt1_threshold_.thB[cnt]);
        }
        fprintf (fp,"\n");
        
      }
      break;
      
      
    case HR_TYPE_NOTICE:
      ptr_text_buf = hpkt1_notice_.text;
      fprintf (fp, "HPKT1 crate %2d NOTICE type %2d %4.4d/%3.3d %2.2d:%2.2d:%2.2d.%3.3d %s\n",
       hpkt1_raw_.pktHdr_crate, hpkt1_notice_.type,
       hpkt1_notice_.year, hpkt1_notice_.day, hpkt1_notice_.hour,
       hpkt1_notice_.min, hpkt1_notice_.sec, hpkt1_notice_.msec,
       ptr_text_buf);
      break;


    case HR_TYPE_REMOTE:
      fprintf (fp, "HPKT1 crate %2d REMOTE tag ", hpkt1_raw_.pktHdr_crate);
      i = 0;
      while((i<8) && (hpkt1_remote_.tag[i] != '\0')) {
        fprintf (fp, "%c", hpkt1_remote_.tag[i]);
        i++;
      }
      ptr_text_buf = hpkt1_remote_.text;
      fprintf (fp, " command %s \n", ptr_text_buf);
      break;
      
      
    case HR_TYPE_CALIB:
      fprintf (fp, "HPKT1 crate %2d CALIB type %1d count %4d ampl %5d width %5d period %5d delay %6d\n",
               hpkt1_raw_.pktHdr_crate, hpkt1_calib_.type,
               hpkt1_calib_.count, hpkt1_calib_.ampl,
               hpkt1_calib_.width, hpkt1_calib_.period,
               hpkt1_calib_.delay*50);
      if(*long_output == 1) {
        fprintf (fp,"means:\n");
        for(cnt=0;cnt<HR_UNIV_MIRTUBE;cnt++) {
          if(cnt%16 == 0) fprintf(fp, "\n");
          fprintf (fp,"%6d ", hpkt1_calib_.mean[cnt]);
        }
        fprintf (fp,"\nstandard deviations:\n");
        for(cnt=0;cnt<HR_UNIV_MIRTUBE;cnt++) {
          if(cnt%16 == 0) fprintf(fp, "\n");
          fprintf (fp,"%5d ", hpkt1_calib_.sdev[cnt]);
        }
        fprintf (fp,"\n");
      }
      break;
      
      
    case HR_TYPE_MSTAT:
      fprintf (fp, "HPKT1 crate %2d MSTAT sent %6d:%6d rcvd %6d:%6d errors %6d warns %6d lost %6d halts %6d max %6d\n",
               hpkt1_raw_.pktHdr_crate, hpkt1_mstat_.sent,
               hpkt1_mstat_.resent, hpkt1_mstat_.rcvd,
               hpkt1_mstat_.rercvd, hpkt1_mstat_.errs,
               hpkt1_mstat_.warns, hpkt1_mstat_.lost,
               hpkt1_mstat_.halts, hpkt1_mstat_.maxMsgs);
      break;
      
      
    case HR_TYPE_VOLTS:
      fprintf (fp, "HPKT1 crate %2d VOLTS min %3d obVer %1d\n",
               hpkt1_raw_.pktHdr_crate, hpkt1_volts_.minute,
               hpkt1_volts_.obVer);
      if(*long_output == 1) {
        fprintf (fp, " Ommatidial Board Readout:\n");
        for(cnt=0;cnt<16;cnt++) {
          fprintf (fp, "board %2d p12v %5d p05v %5d n12v %5d n05v %5d tdcRef %5d temp %5d thRef %5d gnd %5d\n",
                   cnt, hpkt1_volts_.ob_p12v[cnt], hpkt1_volts_.ob_p05v[cnt],
                   hpkt1_volts_.ob_n12v[cnt], hpkt1_volts_.ob_n05v[cnt],
                   hpkt1_volts_.ob_tdcRef[cnt], hpkt1_volts_.ob_temp[cnt],
                   hpkt1_volts_.ob_thRef[cnt], hpkt1_volts_.ob_gnd[cnt]);
        }
        fprintf (fp, "\nGarbage Board Readout:\n");
        fprintf (fp,
          "temp %5d p12v %5d n12v %5d p05v %5d s05v %5d lemo1 %5d anlIn %5d\n",
                 hpkt1_volts_.garb_temp, hpkt1_volts_.garb_p12v,
                 hpkt1_volts_.garb_n12v, hpkt1_volts_.garb_p05v,
                 hpkt1_volts_.garb_s05v, hpkt1_volts_.garb_lemo1,
                 hpkt1_volts_.garb_anlIn);
        fprintf (fp,
         "clsVolts %5d clsTemp %5d mirX %5d mirY %5d clsX %5d clsY %5d\n",
                 hpkt1_volts_.garb_clsVolts, hpkt1_volts_.garb_clsTemp,
                 hpkt1_volts_.garb_mirX, hpkt1_volts_.garb_mirY,
                 hpkt1_volts_.garb_clsX, hpkt1_volts_.garb_clsY);
        fprintf (fp, "ns %5d hvSup %5d hvChnl%3d\n\n",
                 hpkt1_volts_.garb_ns, hpkt1_volts_.garb_hvSup,
                 hpkt1_volts_.garb_hvChnl);
        fprintf (fp, "cluster mux ADC channels:\n");
        for(cnt=0;cnt<16;cnt++) {
          fprintf (fp, "%5d ", hpkt1_volts_.cluster[cnt]);
        }
        fprintf (fp, "\n\nhigh voltages:\n");
        if( hpkt1_volts_.obVer == 3 ) {
          for(cnt=0;cnt<REV3_VOLT_CHNLS;cnt++) {
            if( (cnt%16) == 0 ) fprintf(fp, "\n");
            fprintf(fp, " %4d", hpkt1_volts_.hv[cnt]);
          } 
          fprintf(fp, "\n\n");
        } else {
          for(cnt=0;cnt<REV4_VOLT_CHNLS;cnt++) {
            fprintf(fp, " %4d", hpkt1_volts_.hv[cnt]);
          } 
          fprintf(fp, "\n\n");
        }        
      }
      break;
      
      
    case HR_TYPE_BOARDID:
      if(*long_output != 1) {
        fprintf (fp, "HPKT1 crate %2d BOARD_ID ver %1d\n",
                 hpkt1_raw_.pktHdr_crate, hpkt1_boardid_.version);
      } else {
        fprintf (fp, "HPKT1 crate %2d BOARD_ID ver %1d cpu %4x spare %4x ppg %4x trig %4x garb %4x\n\n",
                 hpkt1_raw_.pktHdr_crate, hpkt1_boardid_.version,
                 hpkt1_boardid_.cpu, hpkt1_boardid_.spare,
                 hpkt1_boardid_.ppg, hpkt1_boardid_.trig,
                 hpkt1_boardid_.garb);
        fprintf (fp, "ommatidial board:\n");
        for(cnt=0;cnt<16;cnt++) {
          fprintf(fp, "%6d ", hpkt1_boardid_.ob[cnt]);
        }
        fprintf(fp, "\n\n");
      }
      break;
    } 
  return 0;
}  

#ifndef _rusdpass0_const_h_
#define _rusdpass0_const_h_


#define NRAWFILESPT 1000      // Max. number of raw data files per tower
#define DOUT_NAME_LEN  0x400                 // max. length of output directory
#define ASCII_NAME_LEN (DOUT_NAME_LEN+0x100) // max. length of ascii file name
#define ASCII_LINE_LEN 0x400                 // max. length of the line acquired from ascii file

#define TMATCH_USEC 100 // Events are time-matching if they are separated by TMATCH_USEC microseconds (default)
#define DUP_USEC 200 // Events occuring within 200uS or less are considered duplicate triggers (default)
#define NRUSDRAW_TOWER 128 // max number of events for BR, LR, SK in buffer
#define NRUSDRAW_TMATCHED 384 // max number of tmatched events in buffer
#define NSDMON_TOWER 8 // max number of mon cycles for BR, LR, SK in buffer
#define NSDMON_TMATCHED 24 // max number of tmatched mon cycles in buffer
#define NMIN_GOOD_EVENTS 10 // minimum number of well calibrated events to call it a successfull day

// Return values for the parser
#define READOUT_FAILURE -1 // Something is wrong and the raw data should be carefully checked
#define READOUT_ENDED 0  // Data file ended before the date ended
#define READOUT_EVENT 1  // Read out an event
#define READOUT_MON 2    // Read out a monitoring cycle
#define MAXNGPST 600     // Maximum number of saved trigger times times in a buffer
// number of elements we would like to have in the GPS time recovery buffer
#define N_TLINE_RECOVERY 2

// date of the transition from data set 2 to data set 3 
// (data set 1 has same layout as data set 2)  
// from this day and until DS3 detector layout changed we have to use
// data set 3 detector layouts.
#define DS2_TO_DS3 91111

#endif

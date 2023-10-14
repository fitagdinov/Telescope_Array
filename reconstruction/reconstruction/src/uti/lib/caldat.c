void caldat(double julian, int *mm, int *id, int *iyyy) {
  int ja, jalpha, jb, jc, jd, je;

  if ( julian >= 2299161.0 ) {
    jalpha = (int)( ( ( julian - 1867216.0 ) - 0.25 ) / 36524.25 );
    ja = (int)( julian + 1.00 + 0.75 * (double)jalpha );
  } 
  else
    ja = (int) julian;

  jb = (int)( (double)ja + 1524.5 );
  jc = (int)( 6680.0 + ( (float)(jb-2439870)-122.1) / 365.25 );
  jd = (int)( 365.25*(double)jc );
  je = (int)( (double)( jb - jd ) / 30.6001 );

  *id = jb - jd - (int)( 30.6001 * (double)je );
  *mm = je - 1;
  if ( *mm > 12 ) 
    *mm -= 12;

  *iyyy = jc - 4715;

  if ( *mm > 2 ) 
    --(*iyyy);

  if ( *iyyy <= 0 ) 
    --(*iyyy);

}


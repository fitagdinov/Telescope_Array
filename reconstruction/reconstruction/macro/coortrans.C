using namespace TMath;


// Convert year, month, day to julian days since 1/1/2000
int greg2jd(int year, int month, int day)
{
  int a, b, c, e, f;
  int iyear, imonth, iday;
  iyear = year;
  imonth = month;
  iday = day;
  if (imonth <= 2)
    {
      iyear -= 1;
      imonth += 12;
    }
  a = iyear/100;
  b = a/4;
  c = 2-a+b;
  e = (int)floor(365.25 * (double)(iyear+4716));
  f = (int)floor(30.6001 * (imonth+1));
  // Julian days corresponding to midnight since Jan 1, 2000
  return (int) ((double)(c+iday+e+f)-1524.5 - 2451544.5);
}


// C = A cross B
void crossp(double *a, double *b, double *c)
{
  c[0] = a[1]*b[2]-a[2]*b[1];
  c[1] = a[2]*b[0]-a[0]*b[2];
  c[2] = a[0]*b[1]-a[1]*b[0];
}

double dotp(double *a, double *b)
{
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

double normv(double *a, double *b)
{
  int i;
  double vmag;
  vmag=sqrt(dotp(a,a));
  for (i=0;i<3;i++)
    b[i]=a[i]/vmag;
  return vmag;
}



// Only rotates from FD frame to CLF frame
// xclf - 3-vector in CLF frame (INPUT)
// xfd  - 3-vector in FD frame (OUTPUT)
bool rot_fdsite2clf(Int_t fdsiteid, Double_t *xfd, Double_t *xclf)
  {
    Int_t i, j;
    Double_t x[3];
    // want to do it so that
    // can use same vector as input and output
    for (i = 0; i < 3; i++)
      x[i] = xfd[i];
    if (fdsiteid == 0)
      {
        for (i = 0; i < 3; i++)
          {
            xclf[i] = 0.0;
            for (j = 0; j < 3; j++)
              xclf[i] += br2clf_mat[i][j] * x[j];
          }
      }
    else if (fdsiteid == 1)
      {
        for (i = 0; i < 3; i++)
          {
            xclf[i] = 0.0;
            for (j = 0; j < 3; j++)
              xclf[i] += lr2clf_mat[i][j] * x[j];
          }
      }
    else if (fdsiteid == 2)
      {
        for (i = 0; i < 3; i++)
          {
            xclf[i] = 0.0;
            for (j = 0; j < 3; j++)
              xclf[i] += md2clf_mat[i][j] * x[j];
          }
      }
    else
      {
        fprintf(stderr,"fdsiteid = %d is not supported\n", fdsiteid);
        for (i = 0; i < 3; i++)
          xfd[i] = 0.0;
        return false;
      }
    return true;
  }


// Only rotates from CLF to FD frame
// xclf - 3-vector in CLF frame (INPUT)
// xfd  - 3-vector in FD frame (OUTPUT)
bool rot_clf2fdsite(Int_t fdsiteid, Double_t *xclf, Double_t *xfd)
  {
    Int_t i, j;
    Double_t x[3];
    // want to do it so that
    // can use same vector as input and output
    for (i = 0; i < 3; i++)
      x[i] = xclf[i];
    if (fdsiteid == 0)
      {
        for (i = 0; i < 3; i++)
          {
            xfd[i] = 0.0;
            for (j = 0; j < 3; j++)
              xfd[i] += br2clf_mat[j][i] * x[j];
          }
      }
    else if (fdsiteid == 1)
      {
        for (i = 0; i < 3; i++)
          {
            xfd[i] = 0.0;
            for (j = 0; j < 3; j++)
              xfd[i] += lr2clf_mat[j][i] * x[j];
          }
      }
    else if (fdsiteid == 2)
      {
        for (i = 0; i < 3; i++)
          {
            xfd[i] = 0.0;
            for (j = 0; j < 3; j++)
              xfd[i] += md2clf_mat[j][i] * x[j];
          }
      }
    else
      {
        fprintf(stderr, "fdsiteid = %d is not supported\n", fdsiteid);
        for (i = 0; i < 3; i++)
          xfd[i] = 0.0;
        return false;
      }
    return true;
  }


// fdsiteid: 0 - BR, 1 - LR (INPUT)
// xfd[3]  - vector in FD frame, [meters] (INPUT)
// xclf[3] - vector in CLF frame, [meters] (OUTPUT)
// vmag  - (optional) vector magnitude, meters (OUTPUT)
bool fdsite2clf(Int_t fdsiteid, Double_t *xfd, Double_t *xclf, Double_t *vmag = 0)
{
  Double_t x[3];          // temporary vector
  Int_t i,j;
  if (fdsiteid == 0)
    {
      for (i=0; i < 3; i++)
	x[i] = xfd[i] + br_origin_clf[i];
      for (i=0; i<3; i++)
	{
	  xclf[i] = 0.0;
	  for (j=0; j<3; j++)
	    xclf[i] +=  br2clf_mat[i][j] * x[j];
	}
    }
  else if (fdsiteid == 1)
    {
      for (i=0; i < 3; i++)
	x[i] = xfd[i] + lr_origin_clf[i];
      for (i=0; i<3; i++)
	{
	  xclf[i] = 0.0;
	  for (j=0; j<3; j++)
	    xclf[i] +=  lr2clf_mat[i][j] * x[j];
	}
    }
  else if (fdsiteid == 2)
    {
      for (i=0; i < 3; i++)
	x[i] = xfd[i] + md_origin_clf[i];
      for (i=0; i<3; i++)
	{
	  xclf[i] = 0.0;
	  for (j=0; j<3; j++)
	    xclf[i] +=  md2clf_mat[i][j] * x[j];
	}
    }
  else
    {
      fprintf (stderr, "fdsiteid = %d is not supported\n",fdsiteid);
      for (i=0; i< 3; i++)
	xclf[i] = 0.0;
      if (vmag)
	(*vmag) = 0.0;
      return false;
    }
  if (vmag)
    {
      (*vmag) = 0.0;
      for (i=0; i<3; i++)
	(*vmag) += xclf[i]*xclf[i];
      (*vmag) = sqrt ((*vmag));
    }
  return true;
}


// fdsiteid: 0 - BR, 1 - LR (INPUT)
// xclf[3] - vector in CLF frame, [meters], (INPUT)
// xfd[3]  - vector in FD frame, [meters], (OUTPUT)
// vmag  - (invariant) vector magnitude, meters (OUTPUT)
bool clf2fdsite(Int_t fdsiteid, Double_t *xclf, Double_t *xfd, Double_t *vmag=0)
{
  Double_t x[3];          // temporary vector
  Int_t i,j;
  if (fdsiteid == 0)
    {
      for (i=0; i < 3; i++)
	x[i] = xclf[i] - br_origin_clf[i];
      // want to use a transpose of FD to CLF rotation matrix
      // so that we can use this transpose to rotate CLF-frame vector
      // into FD-frame vector
      for (i=0; i<3; i++)
	{
	  xfd[i] = 0.0;
	  for (j=0; j<3; j++)
	    xfd[i] +=  br2clf_mat[j][i] * x[j];
	}
    }
  else if (fdsiteid == 1)
    {
      for (i=0; i < 3; i++)
	x[i] = xclf[i] - lr_origin_clf[i];
      for (i=0; i<3; i++)
	{
	  xfd[i] = 0.0;
	  for (j=0; j<3; j++)
	    xfd[i] +=  lr2clf_mat[j][i] * x[j];
	}
    }
  else if (fdsiteid == 2)
    {
      for (i=0; i < 3; i++)
	x[i] = xclf[i] - md_origin_clf[i];
      for (i=0; i<3; i++)
	{
	  xfd[i] = 0.0;
	  for (j=0; j<3; j++)
	    xfd[i] +=  md2clf_mat[j][i] * x[j];
	}
    }
  else
    {
      fprintf (stderr, "fdsiteid = %d is not supported\n",fdsiteid);
      for (i=0; i< 3; i++)
	xfd[i] = 0.0;
      if (vmag)
	(*vmag) = 0.0;
      return false;
    }
  if (vmag)
    {
      (*vmag) = 0.0;
      for (i=0; i<3; i++)
	(*vmag) += xfd[i]*xfd[i];
      (*vmag) = sqrt ((*vmag));
    }
  return true;
}

void conv2vect(Double_t alt, Double_t azm, Double_t *v)
{
  v[0] = cos(alt)*cos(azm);
  v[1] = cos(alt)*sin(azm);
  v[2] = sin(alt);
}


/* $Source: /hires_soft/uvm2k/uti/coords.c,v $
 * $Log: coords.c,v $
 * Revision 1.1  2000/09/14 21:13:44  jeremy
 * Initial revision
 *
*/

#include <math.h>
#include <string.h>
#include "dst_std_types.h"
#include "geoh_dst.h"
#include "astro.h"

typedef struct { char *name;
                 float a0, e0, e1, omega0, omega1;
                 float l0, l1, i0, i1, m0, m1; } planet_data;
typedef struct { char *name; int unob;
                 float alph, delt; } star_data;

/**************************************************************/
/* Private data                                               */
/**************************************************************/

static planet_data pdata[N_PLANETS] =
{
  {"Venus", 0.7233316, 0.00682069, -4.774E-05, 1.3226043, 0.0157053,
   5.9824136, 1021.3529, 0.0592300, 1.75545E-05, 3.710626, 1021.3283},
  {"Mars", 1.5236883, 0.09331290, 9.2064E-05, 0.85148404, 0.0134563,
   5.1266836, 334.08561, 0.0322944, -1.17810E-05, 5.576661, 334.05348},
  {"Jupiter", 5.202561, 0.04833475, 1.6418E-04, 1.7356150, 0.0176371,
   4.1547433, 52.99347, 0.0228418, -9.94157E-05, 3.932721, 52.965368},
  {"Saturn", 9.554747, 0.05589232, -3.455E-04, 1.9685641, 0.0152401,
   4.6524260, 21.354276, 0.0435027, -6.83977E-05, 3.062463, 21.320095}
};

static star_data sdata[N_STARS] = {
  {"21 And", 1, 0.03658, 0.50773},    {"27 Cas", 1, 0.2474, 1.0597},
  {"Algol", 1, 0.82103, 0.7148},      {"45 Per", 1, 1.03782, 0.6983},
  {"Aldebaran", 1, 1.2039, 0.28814},  {"Rigel", 0, 1.3724,-0.14315},
  {"Capella", 1, 1.38179, 0.8028},    {"Bellatrix", 0, 1.41865, 0.1108},
  {"El Nath", 0, 1.42368, 0.4993},    {"Psr 0525+21", 1, 1.432, 0.384},
  {"34 Ori", 0, 1.44865, -0.00497},   {"Crab Nebula", 1, 1.457, 0.385},
  {"44 Ori", 1, 1.46353,-0.10315},    {"Alnilam", 0, 1.46695,-0.02098},
  {"Alnitak", 0, 1.4868,-0.0339},     {"Saiph", 0, 1.51735,-0.16877},
  {"Betelgeuse", 1, 1.54971,0.12928}, {" 2 Cma", 0, 1.66977,-0.3134},
  {"Sirius", 0, 1.76779,-0.29175},    {"Adhara", 0, 1.82656,-0.50566},
  {"Castor", 1, 1.98349, 0.5566},     {"Procyon", 0, 1.0041, 0.0912},
  {"Pollux", 1, 2.0303, 0.48915},     {"Regulus", 0, 2.6545, 0.2089},
  {"Spica", 0, 3.51328, -0.1948},     {"Alkaid", 0, 3.6108, 0.8607},
  {"Arcturus", 1, 3.73348, 0.3348},   {"Antares", 1, 4.3171, -0.46132},
  {"Vega", 0, 4.87354, 0.6769},       {"Altair", 0, 5.1957, 0.15478},
  {"Cygnus X-3", 1, 5.380, 0.713},    {"Deneb", 0, 5.4167, 0.7903},
  {"Fomalhaut", 1, 6.0111, -0.517},   {"Nunki", 0, 4.95346, -0.45896},
  {" 7 Sco", 1, 4.19017,-0.39482},    {"13 Genib", 1, 4.3509,-0.18443},
  {"Alioth", 1, 3.37734, 0.97668},    {"Algenib", 1, 0.05774, 0.265},
  {"Menkalinan", 1, 1.5687, 0.78448}, {"Alhanah", 1, 1.7353, 0.2862},
  {"Atik", 1, 1.02158, 0.55647},      {"Alphecca", 1, 4.07833, 0.4663},
  {"zeta Pup", 0, 2.10774, -0.6982},  {"kappa Sco", 1, 4.6341, -0.6812},
  {"23 Sco", 1, 4.3443, -0.4925},     {"Aludra", 1, 1.93772,-0.49398},
  {"Graffias", 1, 4.2125,-0.34567},   {" 6 Sco", 1, 4.18043,-0.45577},
  {"Pherkad", 1, 4.028, 1.257},       {"Her X1", 0, 4.4358, 0.6201},
  {"Psr 1937+21", 0, 5.137, 0.3747},  {"SS433", 0, 5.014, 0.0855}
};
  
/**************************************************************/
/* Utility functions                                          */
/**************************************************************/

/* static int subc(int it) */
/* { */
/*   return (it >> 4 & 0x0C) + (it >> 2 & 0x03); */
/* } */

/* static int subt(int it) */
/* { */
/*   return (it >> 2 & 0x0C) + (it & 0x03); */
/* } */

/* static int tube(int ic, int it) */
/* { */
/*   return (ic << 4 & 0xC0) + (it << 2 & 0x30) + */
/*          (ic << 2 & 0x0C) + (it & 0x03); */
/* } */

/**************************************************************/
/* Planet and star names. starName also returns a flag        */
/* indicating whether the star is visable in the UV.          */
/**************************************************************/

char *planetName(int n)
{
  return (n >= 0 && n < N_PLANETS) ? pdata[n].name : NULL;
}

int planet_name_(int *n, char *name)
{
  char *pn = planetName(*n);
  if (pn) strcpy(name, pn);
  return strlen(pn);
}

char *starName(int n, int *uvob)
{
  char *name = NULL;
  if (n >= 0 && n < N_STARS)
  {
    if (uvob != NULL) *uvob = !sdata[n].unob;
    name = sdata[n].name;
  }
  return name;
}

int star_name_(int *n, char *name, int *unob)
{
  char *sn = starName(*n, unob);
  if (sn) strcpy(name, sn);
  return strlen(sn);
}

/**************************************************************/
/* Compute direction cosines toward an object at equatorial   */
/* coordinates ra, dec (radians, equinox of date) at local    */
/* sidereal time st (radians).                                */
/*           x: east;   y: north;   z: upwards                */
/**************************************************************/
void objectCosines(float ra, float dec, float st,
		   float *x, float *y, float *z)
{
  float sinalt, cosalt;
  float a, ha = st - ra;
  a = atan2(sin(ha), cos(ha) * sin(DUGLAT)
	    - sin(dec) * cos(DUGLAT) / cos(dec));
  sinalt = sin(DUGLAT) * sin(dec) + cos(DUGLAT) * cos(dec) * cos(ha);
  cosalt = sqrt(1.0 - sinalt * sinalt);
  *z = sinalt; *y = -cos(a) * cosalt; *x = -sin(a) * cosalt;
}

void object_cosines_(float *ra, float *dec, float *st,
		     float *x, float *y, float *z)
{
  objectCosines(*ra, *dec, *st, x, y, z);
}

/**************************************************************/
/* Adjust local coordinates (x, y, z) of object being viewed  */
/* to account approximatly for the effects of refraction in   */
/* Earths atmosphere. This is not particularly accurate       */
/**************************************************************/
void refractionAdjust(float *x, float *y, float *z)
{
  double s, zen, zenr;
  float xx = *x, yy = *y, zz = *z;
  if (zz > 0.999 || zz < -0.03) return;
  zen = atan2(sqrt(1.0 - zz * zz), zz);
  if (zen <= 1.5)
  { /* zenith angle < 86 degrees */
    int i;
    for (zenr = zen, i = 0; i < 2; ++i)
      zenr = zen - (0.00029 * sin(zenr) / cos(zenr));
  }
  else
  { /* very rough approximation _somebody_ made up */
    zenr = zen - (0.0034 + 1.25 * (zen - 1.5) * (zen - 1.5));
  }
  s = sin(zenr) / sqrt(xx * xx + yy * yy);
  *z = cos(zenr); *y = yy * s; *x = xx * s;
}

void refraction_adjust_(float *x, float *y, float *z)
{
  refractionAdjust(x, y, z);
}

/**************************************************************/
/* Account approximatly for the effects of parallax on the    */
/* position of the moon. It is assumed that the horizontal    */
/* parallax is 57.4 arcmin.                                   */
/**************************************************************/
void parallaxAdjust(float *x, float *y, float *z)
{
  double zen, szen;
  float xx = *x, yy = *y, zz = *z;
  szen = sqrt(1.0 - zz * zz);
  zen = atan2(szen, zz) + 0.0167 * szen;
  szen = sin(zen) / szen;
  *x = xx * szen; *y = yy * szen; *z = cos(zen);
}

void parallax_adjust_(float *x, float *y, float *z)
{
  parallaxAdjust(x, y, z);
}

/**************************************************************/
/* Finds tube(s) that contain the coordinates (x, y, z) and   */
/* returns the number of tube found (<= maxt).                */
/**************************************************************/
/*
 * findTubes will need to be rewritten with something replacing geoh_cent_
 * DRB 20081007
 */
/* int findTubes(float x, float y, float z, int maxt, */
/* 	       int mir[], int sclu[], int stub[]) */
/* { */
/*   static int geo_init = 0; */
/*   int n, im, it; */
/*   if (z < 0.0 || z > 0.309016994) /\* below zero or above 18 deg *\/ */
/*     return 0; */
/*   if (geo_init == 0) */
/*   { /\* initialize mirror geometery data *\/ */
/*     geoh_cent_(); */
/*     geo_init = 1; */
/*   } */
/*   for (n = im = 0; im < 22; ++im) */
/*   { */
/*     if ((x * geoh_.xvmir[im] + */
/* 	 y * geoh_.yvmir[im] + */
/* 	 z * geoh_.zvmir[im]) < 0.970295726) /\* < cos(14 deg) *\/ */
/*       continue; */
/*     for (it = 0; it < 256; ++it) */
/*     { */
/*       if ((x * geoh_.xvtube[im][it] + */
/* 	   y * geoh_.yvtube[im][it] + */
/* 	   z * geoh_.zvtube[im][it]) < 0.999925370) /\* < cos(0.7 deg) *\/ */
/* 	continue; */
/*       mir[n] = im + 1; sclu[n] = subc(it) + 1; stub[n] = subt(it) + 1; */
/*       if (++n >= maxt) goto no_more_tubes; */
/*     } */
/*   } */
/*  no_more_tubes: */
/*   return n; */
/* } */

/* int find_tubes_(float *x, float *y, float *z, int *maxt, */
/* 		int *mir, int *sclu, int *stub) */
/* { */
/*   return findTubes(*x, *y, *z, *maxt, mir, sclu, stub); */
/* } */

/**************************************************************/
/* Computes right ascension and declination of a planet at    */
/* time jd (julian day).                                      */
/**************************************************************/
void planetDirection(int p_code, double jd, float *ra, float *dec)
{
  planet_data *pd;
  double t, a, b, l, ecc, i, omega, m, q, v, zeta, deltal, deltae, e;
  double nu, n, d, rplan, u, eclon, eclat, lam, delta, bet;
  float lamsun, rsun;
  int k;

  if (p_code < 0 || p_code >= N_PLANETS) return;
  pd = pdata + p_code;
  t = (jd - 2415020.0) / 36525.0;
  a = pd->a0;
  l = pd->l0 + pd->l1 * t;
  ecc = pd->e0 + pd->e1 * t;
  i = pd->i0 + pd->i1 * t;
  omega = pd->omega0 + pd->omega1 * t;
  m = pd->m0 + pd->m1 * t;
  if (p_code == 2 || p_code == 3) /* Jupiter and Saturn */
  {
    q = 0.73865694 + t * 3.3947608;
    q = 2.0 * M_PI * (q - (int)q);
    v = 0.37397611 + t * 0.11321472;
    v = 2.0 * M_PI * (v - (int)v);
    zeta = 0.07900264 - t * 5.0355339;
    zeta = 2.0 * M_PI * (zeta - (int)zeta);
    if (p_code == 2) /* Jupiter */
    {
      deltal = 0.0057834 * sin(v);
      deltae = (3606.0 * sin(v) - 6764.0 * sin(zeta) * sin(q)
		+ 6074.0 * cos(zeta) * cos(q)) * 1.0e-7;
      b = -0.00035654 * cos(v) + 0.00059404 * cos(zeta) * sin(q)
	+ 0.00065905 * sin(zeta) * cos(q);
    }
    else /* Saturn */
    {
      deltal = -0.014210 * sin(v) - 0.002597 * sin(zeta)
	+ 0.001420 * cos(zeta) * sin(q) + 0.001494 * sin(zeta) * cos(q);
      deltae = (13381.0 * cos(v) + 12415.0 * sin(q)
		+ 26599.0 * cos(zeta) * sin(q)
		- 12696.0 * sin(zeta) * cos(q)) * 1.0e-7;
      b = 0.0013458 * sin(v) - 0.0013234 * sin(zeta) * sin(q)
	- (0.0012669 + 0.0026247 * cos(zeta)) * cos(q);
      a += 0.033629 * cos(zeta);
    }
    l += deltal;
    m += deltal - b / ecc;
    ecc += deltae;
  }
  for (e = m, k = 0; k < 5; ++k)
    e = m + ecc * sin(e);
  nu = 2.0 * atan((sin(e / 2.0) / cos(e / 2.0))
		  * sqrt((1.0 + ecc) / (1.0 - ecc)));
  rplan = a * (1.0 - ecc * cos(e));
  u = l + nu - m - omega;
  eclon = omega + atan2(cos(i) * sin(u), cos(u));
  eclat = sin(u) * sin(i);
  eclat = atan(eclat / sqrt(1.0 - eclat * eclat));
  astroSunmon(jd, 0, &lamsun, NULL, NULL, NULL, &rsun);
  n = rplan * cos(eclat) * sin(eclon - lamsun);
  d = rplan * cos(eclat) * cos(eclon - lamsun) + rsun;
  lam = lamsun + atan2(n, d);
  delta = rplan * sin(eclat);
  delta = sqrt(n * n + d * d + delta * delta);
  bet = rplan * sin(eclat) / delta;
  bet = atan(bet / sqrt(1.0 - bet * bet));
  astroEctoeq(lam, bet, ra, dec);
}

void planet_direction_(int *p_code, double *jd, float *ra, float *dec)
{
  planetDirection(*p_code, *jd, ra, dec);
}

/**************************************************************/
/* Computes right ascension and declination of a fixed star   */
/* at time jd (julian day).                                   */
/**************************************************************/
void starDirection(int s_code, double jd, float *ra, float *dec)
{
  float t, lra, ldec;
  float m = 2.235e-4, n = 9.716e-5;
  if (s_code < 0 || s_code >= N_STARS) return;
  lra = sdata[s_code].alph;
  ldec = sdata[s_code].delt;
  t = (jd - 2451512.5) / 365.242;
  *ra = lra += (m + n * sin(lra) * sin(ldec) / cos(ldec)) * t;
  *dec = ldec + t * n * cos(lra);
}

void star_direction_(int *s_code, double *jd, float *ra, float *dec)
{
  starDirection(*s_code, *jd, ra, dec);
}

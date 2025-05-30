This README contains information about the various map projections that can be used

#Global data

The following projections are tested and appropriate for showing data across the full globe (or smaller):

 - PlateCarree (default, fastest)
 - Aitoff
 - Eckert (I-VI)
 - EqualEarth
 - Hammer
 - InterruptedGoodeHomolosine (non-continuous grid)
 - LambertCylindrical
 - Miller
 - Mollweide
 - Robinson
 - RotatedPole  
 - Sinusoidal

The following projections will work for nearly the full globe, but are truncated at some point:

 - Mercator (limited to >10˚ from poles)
 - ObliqueMercator (limited to >10˚ from projection poles)

 - NorthPolarStereo (limited to ~>-60˚S)
 - SouthPolarStereo (limited to ~<60˚N)
 - Stereographic (limited to >20˚ from antipodal point)

#Hemispheric data

The following projections are tested and appropriate for showing data across a single hemisphere:

 - Geostationary
 - Orthographic 
 - Gnomonic (limited to 80˚ from center point, highly distorted beyond 60˚)
 - LambertConformal (limited depending on user settings)
 - NearsidePerspective (limited depending on user settings)

# Untested/may not work correctly

 - AlbersEqualArea
 - AzimuthalEquidistant
 - EuroPP
 - EquidistantConic
 - UTM
 - LambertAzimuthalEqualArea
 - LambertZoneII
 - OSGB
 - OSNI
 - TransverseMercator


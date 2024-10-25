#=========================================================
# APPAWA: Analytical Partial Pixel Area Weight Assignment |
#                                                         |
# Written by Dr. John Jasper Bekx Sept. 2024 - ...        |
# --------------------------------------------------------|
# NOTE: All routines assume a pixel area of (1 x 1) a.u.² |
#       Therefore, APPAWA calculates a fractional weight, |
#       not an absolute partial area.                     |
# ========================================================
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# class Animal():
#     def __init__(self, race, **kwargs):
#         self.race=race
#         self.extra=race + "rasputin"

# class Friend():
#     def __init__(self, name, **kwargs):
#         self.name=name

# class Dog(Animal, Friend):
#     def __init__(self, race, name, breed):
#         super().__init__(race=race, name=name)
#         self.breed=breed
#         test = super(Animal, self).__getattribute__("extra")
#         print("Hekko", test)

# d = Dog(race="dog", name="Fido", breed="Bernese Mt. Dog")
# # print( d.race, d.name, d.breed, d.extra )
# exit()

# Primordial classes that need no building blocks, only input parameters ==========
class PixelGrid():
    # -----------------------------------------------------------------------------
    # This class defines the square pixel grid and contains all of its properties. |
    # NOTE: Though px_len [μm] is provided, APPAWA's end results are fractional    |
    #       weights (equivalent essentially to px_len=1).                          |
    #------------------------------------------------------------------------------
    def __init__( self, N_px, px_len=75., **kwargs ): # ( self, int, float )
        self.N_px   = N_px   # Defines a square (N_px x N_px)-grid
        self.px_len = px_len # Defines the pixel width (=height) in μm - Default=Eiger2 # TODO: verify with DanMAX

        grid_min = 0
        grid_max = N_px # Normalized to (1 x 1) a.u.² pixels. 
        self.grid_sq = range( grid_min, grid_max+1 ) # endpoint inclusive = N_px+1 points
        # Chose not to use a np.linspace, which uses np.int64 (=slow), instead of native int (=fast)

    @classmethod
    def get_pixelIndex( cls, px_orig, N_px ): # ( cls, (int, int), int )
        # ---------------------------------------------------------------
        # Provides the counting index of a given pixel defined by its    |
        # origin (lower-left corner) location, given as (x,y).           |
        # Example for a 3x3 pixel grid:                                  |
        #                             ___________                        |
        #     e.g.: "0" = (0,0)      |_6_|_7_|_8_|                       |
        #           "5" = (2,1)      |_3_|_4_|_5_|                       |
        #           "7" = (1,2)      |_0_|_1_|_2_|                       |
        #                                                                |
        # ---------------------------------------------------------------
        x_px, y_px = px_orig
        return N_px * y_px + x_px

    @classmethod
    def get_pixelOrigin( cls, px_index, N_px ): # ( cls, int, int )
        # ----------------------------------------
        # Provides the inverse of get_pixelIndex. |
        # ----------------------------------------
        return ( int( px_index%N_px ), int( px_index/N_px ) )

class Circle():
    # -----------------------------------------------------------------------------------
    # This class defines one of the circles of which diffraction rings are comprised of. |
    # Since APPAWA returns area fraction weights, it is assumed that R, x0, and y0 are   |
    # given in units of the px_len.                                                      |
    # The circle origin (x0,y0) is to be given with respect to the pixel-grid origin.    |
    #------------------------------------------------------------------------------------
    def __init__(self, R, x0, y0, **kwargs ): # ( self, float, float, float )
        self.R  = R  # Radius
        self.x0 = x0 # Circle origin - x-coordinate
        self.y0 = y0 # Circle origin - y-coordinate
    
    @classmethod
    def get_circ( cls, x, R, x0, y0 ): # ( cls, float, float, float, float )
        #----------------------------------------------------------------------------
        # Provides the y-coordinates at x of a circle of radius R at origin (x0,y0). |
        # ---------------------------------------------------------------------------
        radic  = R**2. - ( x - x0 )**2.
        if( radic < 0. ): # Don't want complex values or RuntimeWarnings from np.sqrt
            y_upper = np.nan
            y_lower = np.nan
        else:
            y_upper = y0 + radic**( 0.5 )
            y_lower = y0 - radic**( 0.5 )
        return y_upper, y_lower

class AziLine():
    # -----------------------------------------------------------------------------------
    # This class defines one of the lines, defined by the azimuthal angle chi, which     |
    # intersects the point x0, y0 used in the azimuthal binning of diffraction rings.    |
    # Since APPAWA returns area fraction weights, it is assumed x0 and y0 are given in   |
    # units of the px_len.                                                               |
    # The point (x0,y0) is to be given with respect to the pixel-grid origin and         |
    # corresponds to the circle origin (x0, y0).                                         |
    #------------------------------------------------------------------------------------
    def __init__( self, chi, x0, y0, **kwargs ): # ( self, float, float, float )

        if( np.abs( chi ) > 2.*math.pi ):
            chi = chi * ( math.pi / 180. )
            print( "WARNING in creation of object instance AziLine: \n" +
                   "Angle chi is likely in deg; expecting rad. \n" +
                   "Conversion made internally, but give input as rad to suppress this warning." )
            # TODO: might spam warnings for many instances created -> use class variable? -> How?
        
        self.chi = chi # Angle between positive x-direction and the line (+ = counter-clockwise)
        self.x0  = x0  # Point around which the line rotates with the angle chi - x-coordinate
        self.y0  = y0  # Point around which the line rotates with the angle chi - y-coordinate

    @classmethod
    def get_line( cls, x, chi, x0, y0 ): # ( cls, float, float, float, float )
        #----------------------------------------------------------------------------------------------
        # Provides the y-coordinates at x of the line rotated about the point (x0,y0) about angle chi. |
        # ---------------------------------------------------------------------------------------------
        if( math.isclose(np.abs(chi), math.pi/2.) ): 
            return np.nan
        else:
            return y0 + ( x - x0 ) * math.tan( chi ) # See "Hesse normal form" for explanation

# Classes that build on primordial classes ==========
class CircleOnGrid(PixelGrid, Circle):
    # -------------------------------------------------------------------------------------
    # This class considers one instance of the class PixelGrid and one of the class Circle | TODO: doesn't really make any instances, only initializes attributes
    # and calculates where the intersections are between the two and to which pixel these  |
    # belong to. For each pixel the fractional area weight enclosed in the circle is also  |
    # calculated. The intersections and area weights are stored in separate dictionaries.  |
    #--------------------------------------------------------------------------------------
    @classmethod
    def get_enclosedAreaMC( cls, px_orig, R, x0, y0, N_MC=10000 ): # ( cls, (int, int), float, float, float, int )
        # ------------------------------------------------------------------------
        # Numerical Monte Carlo integration of the area enclosed by               |
        # a circle at origin (x0, y0) and radius R and                            |
        # a pixel at origin px_orig denoting the lower-left corner of the pixel.  |
        # All lengths are normalized to a pixel length value.                     |
        # Therefore, the result is the fraction of the pixel area being enclosed. |
        # ------------------------------------------------------------------------
        px_x, px_y = px_orig
        x_rand = np.random.uniform(px_x, px_x + 1, N_MC)
        y_rand = np.random.uniform(px_y, px_y + 1, N_MC)
        sum_enclosedByCurve = 0
        for i in range(len(x_rand)):
            if( y_rand[i] >= y0 ):
                enclosed_ByCurve = ( y_rand[i] <= CircleOnGrid.get_circ(x_rand[i], R, x0, y0)[0] ) # [0] = circle y values >= y0
            else:
                enclosed_ByCurve = ( y_rand[i] > CircleOnGrid.get_circ(x_rand[i], R, x0, y0)[1] )  # [1] = circle y values < y0
            sum_enclosedByCurve += enclosed_ByCurve
        Area_px = 1.**2. # Assuming normalized pixel length throughout
        Area_MC = sum_enclosedByCurve * Area_px / N_MC
        return Area_MC
    
    @classmethod
    def Integral_curvedSegment( cls, x, R, x0, y0, px_line, yes_flipXY=False ): # ( cls, float, float, float, int, bool )
        # -----------------------------------------------------------------------------------------
        # Analytical solution of the integral                                                      |   
        #          / int dx' { [ sqrt( R² - (x'-x0)² ) + y0 ] - px_line }; if px_line >= y0,       |
        #   F(x) =                                                                                 |
        #          \ int dx' { (px_line + 1) - [ -sqrt( R² - (x'-x0)² ) + y0 ] }; if px_line < y0, |
        #            = int dx' { sqrt( R² - (x'-x0)² ) + px_line + 1 - y0 }                        |
        # -----------------------------------------------------------------------------------------
        # Allows to calculate the area enclosed by the circle within the pixel defined by          |
        #   px_line = px_origin row                                                                |
        # Area enclosed in pixel between x=x1 and x=x2 is F(x2) - F(x1), where                     |
        #   F(x) = Integral_curvedSegment( x, R, x0, y0, px_line )                                 |
        # -----------------------------------------------------------------------------------------
        # It is possible that the line y=y0 lies somewhere inside the pixel. In that case, the     |
        #   integral should technically be cut into two pieces, because the integrand is different |
        #   for x' values that result in the integrand being above or below y0.                    |
        # Instead of doing this, we implemented the boolean yes_flipXY, which, if True,            |
        #   switches x <-> y. In that case, px_line = px_origin col (make sure you did this), but  |
        #   this way the integrand does not change over the integration range.                     |
        # -----------------------------------------------------------------------------------------
        if( yes_flipXY ): x0, y0 = y0, x0
        yes_bottomHalf = (px_line < int(y0)) 
        term1 = ( (x-x0) * ( R**2. - (x-x0)**2. )**0.5 + R**2. * np.asin( (x-x0) / R ) ) / 2.
        term2 = (-1.)**int(yes_bottomHalf) * x * ( y0 - px_line - int(yes_bottomHalf) )
        return term1 + term2
    
    @classmethod 
    def get_areaSegment( cls, px_orig, intersec1, intersec2, R, x0, y0 ): # ( cls, (int, int), (float, float), (flaot, float), float, float, float )
        # --------------------------------------------------------------
        # Calculates the enclosed area by the circle within the pixel   |
        # between the intersections intersec1 and intersec2.            |
        # --------------------------------------------------------------
        px_x, px_y = px_orig
        x1, y1 = intersec1
        x2, y2 = intersec2

        yes_intersecTwoRows = ( type(y1)==int and type(y2)==int and max(y1,y2)-min(y1,y2)==1 )
        yes_intersecDoubleCol = ( type(x1)==int and type(x2)==int and x1==x2 and np.abs(y2-y1)<1 )
        yes_y0ThroughPixel = ( min(y1,y2) < y0 and max(y1,y2) > y0)
        yes_integrateAfterFlip = yes_intersecTwoRows or yes_intersecDoubleCol or yes_y0ThroughPixel

        yes_integrateNormally = not yes_integrateAfterFlip

        if( yes_integrateNormally ): 
            insec1, insec2 = sorted( [ intersec1, intersec2 ], key=lambda elem:elem[0] ) # sort along x-coordinate
            x1, y1 = insec1
            x2, y2 = insec2
            px_line = px_y
            yes_flip = False
        elif( yes_integrateAfterFlip ): # flip x and y to just use the same integral 
            insec1, insec2 = sorted( [ intersec1, intersec2 ], key=lambda elem:elem[1] ) # sort along y-coordinate
            y1, x1 = insec1 # flip x & y
            y2, x2 = insec2
            px_line = px_x
            yes_flip = True # Needed in the integral to flip x0 and y0
        else: 
            print( "ERROR in get_areaSegment | Impossible case encountered for pixel at origin " + px_orig )
            exit()
        area  = CircleOnGrid.Integral_curvedSegment( x2, R, x0, y0, px_line, yes_flip ) 
        area -= CircleOnGrid.Integral_curvedSegment( x1, R, x0, y0, px_line, yes_flip )

        # If two opposite corners are confined to the circle, we are missing a little rectangular block, added here.
        yes_twoCornersInCirc1 = ((px_x-x0)**2. + (px_y-y0)**2. < R**2.) and ((px_x+1-x0)**2. + (px_y+1-y0)**2. < R**2.)
        yes_twoCornersInCirc2 = ((px_x+1-x0)**2. + (px_y-y0)**2. < R**2.) and ((px_x-x0)**2. + (px_y+1-y0)**2. < R**2.)
        yes_missingBlock = yes_twoCornersInCirc1 or yes_twoCornersInCirc2
        block = 0
        if( yes_missingBlock ):
            yes_rightOfx0 = ( type(x1)!=int )
            xc = x1 * int( yes_rightOfx0 ) + x2 * int( 1-yes_rightOfx0 )
            block = (-1.)**int( 1-yes_rightOfx0 ) * (xc - px_x) + int( 1-yes_rightOfx0 )
        area += block

        return area

    def __init__( self, N_px, R, x0, y0 ): 
        super().__init__( N_px=N_px, R=R, x0=x0, y0=y0 ) # Initialize parent attributes (does not make instances)

        # Identify ROI with circle in it ---
        ROI_ixMin  = int( (x0 - R) )
        ROI_ixMax  = int( (x0 + R) + 0.99 ) # TODO: Could be prone to bugs, e.g. if x0+R = int+0.0001 - Is a tight ROI square absolutely necessary? 
        ROI_x      = range( ROI_ixMin, ROI_ixMax+1 ) 
        self.ROI_x = ROI_x
        ROI_jyMin  = int( (y0 - R) )
        ROI_jyMax  = int( (y0 + R) + 0.99 ) 
        ROI_y      = range( ROI_jyMin, ROI_jyMax+1 )
        self.ROI_y = ROI_y

        # Collecting intersections and the pixels that contain them ---
        px_intersections = {} # Dictionary = { "pixel index with an intersection": [ intersections as tuples ] }
        for i in ROI_x: # Gathering all intersections with columns
            y_U, y_D = Circle.get_circ( x=i, R=R, x0=x0, y0=y0 )

            if( math.isclose( y_U, y_D ) ): # The pixel grid hits the circle exaclty when y_U=y_D 
                continue                    # Ignore this, it will be caught by x_L and x_R

            for y in y_U, y_D: 
                if( np.isnan(y) ): # Skip NaNs
                    continue

                intersection = (i, y) 

                yes_intersecWithRow    = ( math.isclose( y, int(y) ) )    # The intersection is at a pixel row - possible four-way intersection
                yes_intersecCircBottom = ( math.isclose( y, ROI_y[0] ) )  # The intersection grazes the circle bottom
                yes_intersecCircTop    = ( math.isclose( y, ROI_y[-1] ) ) # The intersection grazes the circle top
                yes_intersecFourWay    = yes_intersecWithRow and not ( yes_intersecCircBottom or yes_intersecCircTop )

                for col in 0,1: # Adding pixels right (i) and left (i-1) of the y intersection in one loop
                    if( yes_intersecFourWay ): # Hit a full-on four-way intersection - Add diagonally adjacent pixels
                        yes_TLorBR = ( (i-x0)*(y-y0) < 0 ) # Top left or bottom right quadrant have pixels like / (TR and BL like \)
                        px_origX = i + ( int(-col)*int(yes_TLorBR) + int(col-1)*int(1-yes_TLorBR) )
                        px_origY = int(y) + int(-col)
                    else:
                        yes_needsOriginFix = yes_intersecCircTop # Don't count pixel above the circle top
                        px_origX = i - col
                        px_origY = int(y) - int(yes_needsOriginFix)
                    px_orig = ( px_origX, px_origY )
                    key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                    px_intersections.setdefault(key, []).append( intersection )
            
        for j in ROI_y: # Gathering all intersections with rows
            x_R, x_L = Circle.get_circ( x=j, R=R, x0=y0, y0=x0 ) # Can use same formula if you switch x <-> y.

            if( math.isclose( x_R, x_L ) ): # The pixel grid hits the circle exaclty when x_R=x_L
                continue                    # Ignore this, it was caught by y_U and y_D

            for x in x_L, x_R:
                if( np.isnan(x) ): # Skip NaNs.
                    continue

                yes_intersecWithCol   = ( math.isclose( x, int(x) ) )    # The intersection is at a pixel col - possible four-way intersection
                yes_intersecCircLeft  = ( math.isclose( x, ROI_x[0] ) )  # The intersection grazes the circle left-most tip
                yes_intersecCircRight = ( math.isclose( x, ROI_x[-1] ) ) # The intersection grazes the circle right-most tip
                yes_intersecFourWay   = yes_intersecWithCol and not ( yes_intersecCircLeft or yes_intersecCircRight )

                if( yes_intersecFourWay ): # Caught the four-way intersections in the y-loop above 
                    continue
                
                intersection = (x, j)

                yes_needsOriginFix = yes_intersecCircRight # Don't count pixel to the right of the circle
                for row in 0,1: # Adding the pixels above (j) and below (j-1) of the x intersection in one loop
                    px_origX = int(x) - int(yes_needsOriginFix)
                    px_origY = j - row
                    px_orig = ( px_origX, px_origY )
                    key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                    px_intersections.setdefault(key, []).append( intersection )

        self.px_intersections = px_intersections

        # Collecting areas ---
        px_enclosedArea = {}

        # Pixels in full interior of circle 
        px_indicesSorted = sorted( px_intersections.keys() ) # Ensures order (i, j), (i+1, j), ..., (i, j+1), (i+1, j+1), ... 
        px_withIntersections = [ PixelGrid.get_pixelOrigin( i, N_px=N_px ) for i in px_indicesSorted ]
        for i in range(1,len(px_withIntersections)):
            curr_tup = px_withIntersections[i]
            prev_tup = px_withIntersections[i-1]
            if( curr_tup[1]==prev_tup[1] ): # These are on the same row            
                if( curr_tup[0] == prev_tup[0] + 1 ): # These are adjacent pixels with intersections
                    continue
                else:
                    for j in range(1, curr_tup[0] - prev_tup[0] ): # All of these in between are within the circle
                        px_within = ( prev_tup[0] + j, curr_tup[1] )
                        key = PixelGrid.get_pixelIndex( px_within, N_px )
                        px_enclosedArea[key] = 1. # Area is 1 a.u.²
            else:
                continue

        # Fractional area weights 
        for key, val in px_intersections.items():
            px_orig = PixelGrid.get_pixelOrigin( key, N_px )

            if( len(val) <=1 or len(val) > 4 ): # Number of intersections should be 2, 3 or 4
                raise AssertionError( "ERROR in classifying intersections | Pixel " + str(key) + " has " + str(len(val)) + " intersections." )

            if( len(val)==2 ): # In most cases, there are two intersections
                area = CircleOnGrid.get_areaSegment( px_orig, val[0], val[1], R, x0, y0 )
            elif( len(val)==3 ): # In a rare case, you may be grazing the pixel boundary at a single point. Just ignore this point.
                x_insec = [ i[0] for i in val ]
                y_insec = [ i[1] for i in val ]
                for i in range(len(val)):
                    index1 = int(i/2) # Just looping over (0,1), (0,2), (1,2)
                    index2 = int((i+1)/2)+1
                    if( type(x_insec[index1]) == type(x_insec[index2]) ): 
                        area = CircleOnGrid.get_areaSegment( px_orig, val[index1], val[index2], R, x0, y0 )
                    else:
                        pass
            elif( len(val)==4 ): # Experience has found that four intersections is possible
                x_insec = [ i[0] for i in val ]
                y_insec = [ i[1] for i in val ]
                yes_sameIntersecX = ( len(x_insec)!=len(set(x_insec)) ) # Only one of these two
                yes_sameIntersecY = ( len(y_insec)!=len(set(y_insec)) ) # should be true
                if( yes_sameIntersecX != yes_sameIntersecY):
                    raise AssertionError( "ERROR in classifying intersections | Impossible case encountered for pixel " + str(key) )
                sort_along = int(yes_sameIntersecX) 
                val_ordered = sorted( val, key=lambda elem:elem[ sort_along ] ) # sort along x/y if there are two intersections on the same pixel row/column
                area = 0
                area += CircleOnGrid.get_areaSegment( px_orig, val_ordered[0], val_ordered[1], R, x0, y0 )
                area += CircleOnGrid.get_areaSegment( px_orig, val_ordered[2], val_ordered[3], R, x0, y0 )
                area += int(yes_sameIntersecY) * np.abs( val_ordered[2][0] - val_ordered[1][0] ) + int(yes_sameIntersecX) * np.abs( val_ordered[2][1] - val_ordered[1][1] ) 
            else:
                raise AssertionError( "ERROR in classifying intersections | Pixel " + str(key) + " has " + str(len(val)) + " intersections." )

            if (area > 1.):
                raise AssertionError( "ERROR in calculating fractional area | Area is larger than 1 for pixel " + str(key) )

            px_enclosedArea[key] = area
        
        self.px_enclosedArea = px_enclosedArea

class AziLineOnGrid(PixelGrid, AziLine): 
    #BUG: still ahs pixel idnex assigment outside the pixel grid
    def __init__( self, N_px, chi, x0, y0 ): 
        super().__init__( N_px=N_px, chi=chi, x0=x0, y0=y0 ) # Initialize parent attributes (does not make instances)

        yes_ascendingLine  = (chi >= 0. and chi < math.pi/2.) or (chi >= math.pi and chi < 3.*math.pi/2.) # no chi < 0
        yes_descendingLine = not yes_ascendingLine # BUG is this chi or self.chi? i.e., what if someone fills in 183 deg?

        # Collecting intersections and the pixels that contain them ---
        px_intersections = {} # Dictionary = { "pixel index with an intersection": [ intersections as tuples ] }
        grid_sq = super(PixelGrid, self).__getattribute__("grid_sq")
        for i in grid_sq: # Gathering all intersections with columns
            y = AziLine.get_line( x=i, chi=chi, x0=x0, y0=y0 )

            if( np.isnan(y) ): # Skip NaNs
                continue

            if( y < 0 or y > N_px ): # Skip intersections outside of pixel grid
                continue

            intersection = (i, y) 

            yes_intersecWithRow    = ( math.isclose( y, int(y) ) )    # The intersection is at a pixel row - possible four-way intersection
            yes_intersecGridBottom = ( math.isclose( y, grid_sq[0] ) )  # The intersection hits the pixel grid bottom
            yes_intersecGridTop    = ( math.isclose( y, grid_sq[-1] ) ) # The intersection hits the pixel grid top
            yes_intersecFourWay    = yes_intersecWithRow and not ( yes_intersecGridBottom or yes_intersecGridTop )

            for col in 0,1: # Adding pixels right (i) and left (i-1) of the y intersection in one loop
                if( yes_intersecFourWay ): # Hit a full-on four-way intersection - Add diagonally adjacent pixels
                    px_origX = i + ( int(-col)*int(yes_ascendingLine) + int(col-1)*int(yes_descendingLine) )
                    px_origY = int(y) + int(-col)
                else:
                    yes_needsOriginFix = yes_intersecGridTop # Don't count pixel outside the gird
                    px_origX = i - col
                    px_origY = int(y) - int(yes_needsOriginFix)
                px_orig = ( px_origX, px_origY )
                key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                px_intersections.setdefault(key, []).append( intersection )
            
        for j in grid_sq: # Gathering all intersections with rows
            x = AziLine.get_line( x=j, chi=math.pi/2.-chi, x0=y0, y0=x0 ) # Can use same formula if you switch x <-> y.
            # Switch x <-> equivalent to chi -> pi/2 - chi, x0 -> y0, y0 -> x0

            if( np.isnan(x) ): # Skip NaNs.
                continue

            if( x < 0 or x > N_px ): # Skip intersections outside of pixel grid
                continue

            yes_intersecWithCol   = ( math.isclose( x, int(x) ) )    # The intersection is at a pixel col - possible four-way intersection
            yes_intersecGridLeft  = ( math.isclose( x, grid_sq[0] ) )  # The intersection grazes the circle left-most tip
            yes_intersecGridRight = ( math.isclose( x, grid_sq[-1] ) ) # The intersection grazes the circle right-most tip
            yes_intersecFourWay   = yes_intersecWithCol and not ( yes_intersecGridLeft or yes_intersecGridRight )

            if( yes_intersecFourWay ): # Caught the four-way intersections in the y-loop above 
                continue
            
            intersection = (x, j)

            yes_needsOriginFix = yes_intersecGridRight # Don't count pixel to the right of the circle
            for row in 0,1: # Adding the pixels above (j) and below (j-1) of the x intersection in one loop
                px_origX = int(x) - int(yes_needsOriginFix)
                px_origY = j - row
                px_orig = ( px_origX, px_origY )
                key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                px_intersections.setdefault(key, []).append( intersection )

        self.px_intersections = px_intersections

# Functions designed for development checks ==========
def check_CoGIntersections( N_px, R, x0, y0, N_plt=1000 +1 ): # ( int, float, float, float, int )
    
    # Create an instance ---
    CoG = CircleOnGrid(N_px=N_px, R=R, x0=x0, y0=y0)

    # Collect and print intersections ---
    px_intersections = []
    for key, value in CoG.px_intersections.items():
        pixel = CoG.get_pixelOrigin( int(key), N_px )
        print( "Pixel index = ", key, "at loc ", pixel, "with intersections ", value )
        [ px_intersections.append( value[i] ) for i in range(len(value)) ]
    
    # Plotting ---
    fig, ax = plt.subplots()

    x_plt = np.linspace(0, N_px, num=N_plt, endpoint=True)

    # Pixel grid 
    [ ax.vlines( x=i, ymin=0, ymax=N_px, color="gray", ls="--" ) for i in CoG.grid_sq ]
    [ ax.hlines( y=i, xmin=0, xmax=N_px, color="gray", ls="--" ) for i in CoG.grid_sq ]

    # ROI highlight 
    [ ax.vlines( x=CoG.ROI_x[i], ymin=CoG.ROI_y[0], ymax=CoG.ROI_y[-1], color="k" ) for i in range(len(CoG.ROI_x)) ]
    [ ax.hlines( y=CoG.ROI_y[i], xmin=CoG.ROI_x[0], xmax=CoG.ROI_x[-1], color="k" ) for i in range(len(CoG.ROI_y)) ]

    # Circle 
    y_circle = [ CoG.get_circ( i, R, x0, y0 ) for i in x_plt ]
    y_upper = [ i[0] for i in y_circle ]
    y_lower = [ i[1] for i in y_circle ]
    ax.plot(x_plt, y_upper, color="tab:blue")
    ax.plot(x_plt, y_lower, color="tab:blue")

    # Intersections between grid and circle 
    for i in range(len(px_intersections)):
        ax.plot( px_intersections[i][0], px_intersections[i][1], "x", color="tab:orange" )

    plt.show()
    return 

def check_CoGAreas( N_px, R, x0, y0, yes_MC=False, px_orig=None, N_MC=10000, yes_verbose=False ): # ( int, float, float, float, bool, (int, int), int, bool )

    # Create an instance ---
    t0  = time.time()
    CoG = CircleOnGrid(N_px=N_px, R=R, x0=x0, y0=y0)
    ans = sum( CoG.px_enclosedArea.values() )
    t1  = time.time()
    t_algo = t1 - t0

    # Analytical answer
    t0 = time.time()
    goal = math.pi * R**2.
    t1 = time.time()
    t_anal = t1- t0

    print( "The analytical closed-form answer is: ", goal, " and took ", t_anal, " sec to calculate" )
    print( "The analytical algorithmic answer is: ", ans , " and took ", t_algo, " sec to calculate" )

    # Monte Carlo approximation ---
    if( yes_MC ):
        t0 = time.time()
        CoG = CircleOnGrid(N_px=N_px, R=R, x0=x0, y0=y0) # Create another instance for fair time comparison
        if( px_orig != None ): 
            key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
            area_CoG = CoG.get_enclosedAreaMC[key]
            area_MC  = CircleOnGrid.get_enclosedAreaMC( px_orig=px_orig, R=R, x0=x0, y0=y0, N_MC=N_MC )
            rel = ( 1. - np.abs( area_MC - area_CoG )/ area_MC )*100
            print( "The enclosed area of pixel ", key, " at location ", px_orig, "is:" )
            print( "  - Analytical algorithm: ", area_CoG )
            print( "  - Monte Carlo approxim: ", area_MC, " using ", N_MC, "points" )
            print( "  - Relative accuracy[%]: ", rel )
        else:
            total_MC = 0.
            i=0
            for key in CoG.px_enclosedArea.keys():
                px_orig = PixelGrid.get_pixelOrigin( px_index=key, N_px=N_px )
                total_MC += CircleOnGrid.get_enclosedAreaMC( px_orig=px_orig, R=R, x0=x0, y0=y0, N_MC=N_MC )
                if( yes_verbose ):
                    i+=1
                    print( "Finished with pixel", i, " out of ", len(CoG.px_enclosedArea.keys()) )
            t1 = time.time()
            t_MC = t1 - t0
            print( "The total enclosed area is:" )
            print( "  - Analytical form:      ", goal, " and took ", t_anal, " sec to calculate" )
            print( "  - Analytical algorithm: ", ans, " and took ", t_algo, " sec to calculate" )
            print( "  - Monte Carlo approxim: ", total_MC, " and took ", t_MC, " sec to calculate using ", N_MC, "points" )
    return

def check_ALoGIntersections( N_px, chi, x0, y0, N_plt=1000 +1 ): # ( int, float, float, float, int )
    
    # Create an instance ---
    ALoG = AziLineOnGrid(N_px=N_px, chi=chi, x0=x0, y0=y0)
    
    # Collect and print intersections ---
    px_intersections = []
    for key, value in ALoG.px_intersections.items():
        pixel = ALoG.get_pixelOrigin( int(key), N_px )
        print( "Pixel index = ", key, "at loc ", pixel, "with intersections ", value )
        [ px_intersections.append( value[i] ) for i in range(len(value)) ]
    
    # Plotting ---
    fig, ax = plt.subplots()

    x_plt = np.linspace(0, N_px, num=N_plt, endpoint=True)

    # Pixel grid 
    [ ax.vlines( x=i, ymin=0, ymax=N_px, color="gray", ls="--" ) for i in ALoG.grid_sq ]
    [ ax.hlines( y=i, xmin=0, xmax=N_px, color="gray", ls="--" ) for i in ALoG.grid_sq ]

    # Line center and angle highlight 
    ax.plot(x0, y0, "bo")
    ax.hlines( y=y0, xmin=0, xmax=N_px, color="k", ls="--" )

    # Line 
    y_line = [ ALoG.get_line( i, chi, x0, y0 ) for i in x_plt ]
    ax.plot(x_plt, y_line, color="tab:blue")

    # Intersections between grid and circle 
    for i in range(len(px_intersections)):
        ax.plot( px_intersections[i][0], px_intersections[i][1], "x", color="tab:orange" )

    plt.show()
    return 

N_px = 10
R = 2.
x0 = 5
y0 = 5

# check_CoGIntersections( N_px=N_px, R=R, x0=x0, y0=y0 )
# check_CoGAreas(N_px=N_px, R=R, x0=x0, y0=y0, yes_MC=True, N_MC=1000000, yes_verbose=True)

check_ALoGIntersections( N_px=N_px, chi=math.pi/3., x0=x0, y0=y0 )

exit()

# Functions =======================================================================================
# Pixel-grid related ---
def get_pixelIndex( px_orig, N_px ): # (x, y) - NOT (row, col)
    # ---------------------------------------------------------
    # Provides the counting index of                           |
    # a given pixel defined by its origin (lower-left corner). |
    # Example for a 3x3 pixel grid:                            |
    #                         ___________                      |
    #                        |_6_|_7_|_8_|                     |
    #                        |_3_|_4_|_5_|                     |
    #                        |_0_|_1_|_2_|                     |
    #                        |                                 |
    #                        l-> origin at (0,0)               |
    # ---------------------------------------------------------
    x_px, y_px = px_orig
    return N_px * y_px + x_px

def get_pixelOrigin( px_index, N_px ): # (x, y) - NOT (row, col)
    # ----------------------------------------
    # Provides the inverse of get_pixelIndex. |
    # ----------------------------------------
    return ( int( px_index%N_px ), int( px_index/N_px ) )

# Circle related ---
def circ( x, R, x0, y0 ):
    #------------------------------------------
    # Provides the y-coordinates at x of       |
    # a circle at origin (x0,y0) and radius R. |
    # -----------------------------------------
    y_pos = ( R**2. - ( x - x0 )**2. )**0.5 + y0
    y_neg = -(R**2. - ( x-x0 )**2.)**0.5 + y0
    return y_pos, y_neg

def intersec_x( R, x0, y0, px_y ): # px_y = integer denoting the row of the lower-left corner of the pixel
    #----------------------------------------------------
    # Provides the x-coordinates of the intersections of | 
    # the circle at origin (x0,y0) and radius R and      |
    # the pixel row defined by px_y.                     |
    # ---------------------------------------------------
    if( np.abs( y0-px_y ) <= R ):
        x1 = -(R**2. - ( px_y - y0 )**2.)**0.5 + x0
        x2 = (R**2. - ( px_y - y0 )**2.)**0.5 + x0
    else:
        x1 = np.nan
        x2 = np.nan
    return x1, x2
    
def intersec_y( R, x0, y0, px_x ): # px_x = integer denoting the col of the lower-left corner of the pixel
    #----------------------------------------------------
    # Provides the y-coordinates of the intersections of | 
    # the circle at origin (x0,y0) and radius R and      |
    # the pixel col defined by px_x.                     |
    # ---------------------------------------------------
    if( np.abs( x0-px_x ) <= R ):
        y1 = -(R**2. - ( px_x - x0 )**2.)**0.5 + y0
        y2 = (R**2. - ( px_x - x0 )**2.)**0.5 + y0
    else:
        y1 = np.nan
        y2 = np.nan
    return y1, y2
    
# Enclosed-area related ---
def Area_enclosedPixelMC( px_orig, R, x0, y0, N_MC=10000 ): # px_orig = (x, y) 
    # ------------------------------------------------------------------------
    # Numerical Monte Carlo integration of the area enclosed by               |
    # a circle at origin (x0, y0) and radius R and                            |
    # a pixel at origin px_orig denoting the lower-left corner of the pixel.  |
    # All lengths are normalized to a pixel length value.                     |
    # Therefore, the result is the fraction of the pixel area being enclosed. |
    # ------------------------------------------------------------------------
    px_x, px_y = px_orig
    x_rand = np.random.uniform(px_x, px_x + 1, N_MC)
    y_rand = np.random.uniform(px_y, px_y + 1, N_MC)
    sum_enclosedByCurve = 0
    for i in range(len(x_rand)):
        if( y_rand[i] >= y0 ):
            enclosed_ByCurve = ( y_rand[i] <= circ(x_rand[i], R, x0, y0)[0] ) # [0] = circle y values >= y0
        else:
            enclosed_ByCurve = ( y_rand[i] > circ(x_rand[i], R, x0, y0)[1] )  # [1] = circle y values < y0
        sum_enclosedByCurve += enclosed_ByCurve
    Area_px = 1.**2. # Assuming normalized pixel length throughout
    Area_MC = sum_enclosedByCurve * Area_px / N_MC
    return Area_MC

def Integral_curvedSegment( x, R, x0, y0, px_line, yes_flipXY=False ):
    # -------------------------------------------------------------------------------
    # Note: for this explanation, pretend px_line = px_y = px_origin row (*)         |
    # Analytical solution of the integral                                            |   
    # int dx [ sqrt( R² - (x-x0)² ) + (y0 - px_line) ]                               |  
    # ( + constant omitted )                                                         |
    # Area enclosed in pixel at (px_x, px_y) between x=x1 and x=x2 is I(x2) - I(x1), |
    # where I(x) = Integral_curvedSegment( x, R, x0, y0, px_line )                   |
    #                                                                                |
    # (*) yes_flipXY=True calculates the area for a pixel that is difficult to       |
    # handle as is, but which is easier if one does x <-> y.                         |
    # In that case px_line = px_x = px_origin col, but the integral stays the same.  |
    # -------------------------------------------------------------------------------
    # x0, y0 = x0 * int(1-int(yes_flipXY)) + y0 * int(yes_flipXY), x0 * int(yes_flipXY) + y0 * int(1-int(yes_flipXY))
    if( yes_flipXY ):
        x0, y0 = y0, x0
    yes_bottomHalf = (px_line < int(y0)) 
    term1 = ( (x-x0) * ( R**2. - (x-x0)**2. )**0.5 + R**2. * np.asin( (x-x0) / R ) ) / 2.
    term2 = (-1.)**int(yes_bottomHalf) * x * ( y0 - px_line - int(yes_bottomHalf) )
    return term1 + term2

def get_areaSegment( px_orig, intersec1, intersec2, R, x0, y0 ):
    px_x, px_y = px_orig
    x1, y1 = intersec1
    x2, y2 = intersec2

    yes_intersecTwoRows = ( type(y1)==int and type(y2)==int and max(y1,y2)-min(y1,y2)==1 )
    yes_intersecDoubleCol = ( type(x1)==int and type(x2)==int and x1==x2 and np.abs(y2-y1)<1 )
    yes_y0ThroughPixel = ( min(y1,y2) < y0 and max(y1,y2) > y0)
    yes_integrateAfterFlip = yes_intersecTwoRows or yes_intersecDoubleCol or yes_y0ThroughPixel

    yes_integrateNormally = not yes_integrateAfterFlip

    if( yes_integrateNormally ): 
        insec1, insec2 = sorted( [ intersec1, intersec2 ], key=lambda elem:elem[0] ) # sort along x-coordinate
        x1, y1 = insec1
        x2, y2 = insec2
        px_line = px_y
        yes_flip = False
    elif( yes_integrateAfterFlip ): # flip x and y to just use the same integral 
        insec1, insec2 = sorted( [ intersec1, intersec2 ], key=lambda elem:elem[1] ) # sort along y-coordinate
        y1, x1 = insec1 # flip x & y
        y2, x2 = insec2
        px_line = px_x
        yes_flip = True # Needed in the integral to flip x0 and y0
    else: 
        print( "ERROR in get_areaSegment | Impossible case encountered for pixel at origin " + px_orig )
    area = Integral_curvedSegment( x2, R, x0, y0, px_line, yes_flip ) - Integral_curvedSegment( x1, R, x0, y0, px_line, yes_flip )

    # If two opposite corners are confined to the circle, we are missing a little rectangular block, added here.
    yes_twoCornersInCirc1 = ((px_x-x0)**2. + (px_y-y0)**2. < R**2.) and ((px_x+1-x0)**2. + (px_y+1-y0)**2. < R**2.)
    yes_twoCornersInCirc2 = ((px_x+1-x0)**2. + (px_y-y0)**2. < R**2.) and ((px_x-x0)**2. + (px_y+1-y0)**2. < R**2.)
    yes_missingBlock = yes_twoCornersInCirc1 or yes_twoCornersInCirc2
    block = 0
    if( yes_missingBlock ):
        yes_rightOfx0 = ( type(x1)!=int )
        xc = x1 * int( yes_rightOfx0 ) + x2 * int( 1-yes_rightOfx0 )
        block = (-1.)**int( 1-yes_rightOfx0 ) * (xc - px_x) + int( 1-yes_rightOfx0 )
    area += block

    return area

# Checking test cases ---
def TestCase(c): #returns (R, x0, y0)
    if( c=="A" ):
        R = 2.
        x0 = 5.
        y0 = 5.
    elif( c=="B" ):
        R = 2.
        x0 = 5.5
        y0 = 5.
    elif( c=="C" ):
        R = 2.
        x0 = 5.
        y0 = 5.5
    elif( c=="D" ):
        R = 2.
        x0 = 5.5
        y0 = 5.5
    elif( c=="E" ):
        R = 2.5
        x0 = 5.
        y0 = 5.
    elif( c=="F" ):
        R = 2.5
        x0 = 5.5
        y0 = 5.
    elif( c=="G" ):
        R = 2.5
        x0 = 5.
        y0 = 5.5
    elif( c=="H" ):
        R = 2.5
        x0 = 5.5
        y0 = 5.5
    elif( c=="I" ):
        R = math.pi
        x0 = 5.1111
        y0 = 5.1415
    return (R, x0, y0)

# Azimuthal line related ---
def Line( x, chi, x0, y0 ):
    if( math.isclose(np.abs(chi), math.pi) ): 
        return np.nan
    else:
        return y0 - ( x - x0 ) / math.tan( chi-math.pi/2. )

def intersec_LineX( chi, x0, y0, px_y ): # px_y = integer denoting the row of the lower-left corner of the pixel
    # Same as intersec_LineY with y0 -> x0, px_x -> px_y, x0 -> y0, chi -> pi/2 - chi
    if( math.isclose(np.abs(chi), math.pi) ): # If the Line is vertical, it intersects the row px_y at px_y
        return px_y
    else:
        return x0 - ( px_y - y0 ) * math.tan( chi-math.pi/2. ) 

def intersec_LineY( chi, x0, y0, px_x ): # px_x = integer denoting the col of the lower-left corner of the pixel
    if( math.isclose(np.abs(chi), math.pi) ): # If the Line is vertical, it intersects the col px_x either never or inf many times
        return np.nan
    else:
        return y0 - ( px_x - x0 ) / math.tan( chi-math.pi/2. )

# =================================================================================================

# Parameters ======================================================================================
# Pixel grid ---
N_px     = 10 # = (N_px x N_px) grid 
px_len   = 1. # Pixel size (width=height) - mostly ignored
grid_min = 0
grid_max = N_px # Normalized to 1 x 1 (a.u.²) pixels. 
grid_sq  = range( grid_min, grid_max+1 ) # endpoint inclusive = N_px + 1 points
# grid_sq = np.linspace( grid_min, grid_max, num=N_px+1, endpoint=True, dtype=int ) # linspace uses np.int64 (=slow) instead of native int (=fast)

# Circle ---
# R  = math.pi / px_len # Circle radius in pixel units
# x0 = 5.1111 / px_len  # N_px * px_len / 2. # Circle origin 
# y0 = 10-5.1415 / px_len  # N_px * px_len / 2. 

R, x0, y0 = TestCase("E")

# Azimuthally defined lines --- 
chi = math.pi / 4.12 # Defined according to mathematicians' conventions.

# Main ============================================================================================
t0 = time.time()

# Identify ROI with circle in it ---
ROI_ixMin = int( (x0 - R) )
ROI_ixMax = int( (x0 + R) + 0.99 ) # TODO: Could be prone to bugs - Is a tight ROI square absolutely necessary? 
ROI_x     = range( ROI_ixMin, ROI_ixMax+1 ) # ROI_x = np.linspace( ROI_ixMin, ROI_ixMax, num=ROI_ixMax-ROI_ixMin+1, endpoint=True,dtype=int ) 
ROI_jyMin = int( (y0 - R) )
ROI_jyMax = int( (y0 + R) + 0.99 ) 
ROI_y     = range( ROI_jyMin, ROI_jyMax+1 )# ROI_y = np.linspace( ROI_jyMin, ROI_jyMax, num=ROI_jyMax-ROI_jyMin+1, endpoint=True, dtype=int )

# All pixel-circle intersections within the ROI & identifying to which pixels they correspond to ---
intersec        = [] # All intersection locations as tuples.
px_origIntersec = [] # Pixel origins of pixels that contain the intersections.
dict_pxIntersec = {} # Dictionary = { "pixel index with an intersection": [ intersection1, intersection2 ] }.

for i in ROI_x: # Gathering all intersections with x=integers
    y_D, y_U = intersec_y( R, x0, y0, i )
    if( math.isclose( y_U,y_D ) ): continue # You hit a crossing where y_D=y_U. Ignore this, it will be caught by x_L and x_R
    for y in y_D, y_U: # Classify the intersections
        if( not np.isnan(y) ): # checking y != np.nan leads to some spooky business - there's a difference between numpy nan and "true" nan apparently
            intersec.append( (i, y) ) 
            if( math.isclose( y, int(y) ) and not( math.isclose( y_D, ROI_y[0] ) or math.isclose( y_U, ROI_y[-1] ) ) ): # You hit a full-on four-way crossing. Needs some special treatment.
                yes_TLorBR = ( (i-x0)*(y-y0) < 0 ) # Either top left or bottom right quadrant
                for offset in 0,1:
                    px_orig = ( i + ( int(-offset)*int(yes_TLorBR) + int(offset-1)*int(1-yes_TLorBR) ), int(y) + int(-offset) )
                    px_origIntersec.append( px_orig )
                    px_index = get_pixelIndex( px_orig, N_px )
                    key = str(px_index)
                    dict_pxIntersec.setdefault(key, []).append( (i,y) )
            else:
                yes_needsOriginFix = (y==y_U) and ( math.isclose( y_U, ROI_y[-1]) ) # math.isclose is a bit more consistent than numpy.isclose
                for col in 1,0: # Just doing the pixels left (i-1) and right (i) of the y intersection in one loop
                    px_orig = (i-col,int(y) - int(yes_needsOriginFix))
                    px_origIntersec.append( px_orig )
                    px_index = get_pixelIndex( px_orig, N_px )
                    key = str(px_index)
                    dict_pxIntersec.setdefault(key, []).append( (i,y) )
    
for j in ROI_y:
    x_L, x_R = intersec_x( R, x0, y0, j )
    if( math.isclose( x_R,x_L ) ): continue # You hit a crossing where x_L=x_R. Ignore this, it was caught by y_D and y_U
    for x in x_L, x_R:
        if( not np.isnan(x) ):
            if( math.isclose( x, int(x) ) and not( math.isclose( x_L, ROI_x[0] ) or math.isclose( x_R, ROI_x[-1] ) ) ): continue # caught these in the y-loop above 
            intersec.append( (x, j) )
            yes_needsOriginFix = (x==x_R) and ( math.isclose( x_R, ROI_x[-1] ) )
            for row in 1,0: # Just doing the pixels below (j-1) and above (j) of the x intersection in one loop
                px_origIntersec.append( (int(x) - int(yes_needsOriginFix),j-row) )
                px_index = get_pixelIndex( (int(x) - int(yes_needsOriginFix),j-row), N_px )
                key = str(px_index)
                dict_pxIntersec.setdefault(key, []).append( (x,j) )

# Identifying pixels in full interior of circle ---
px_origIntersec = list( set(px_origIntersec) ) # Removing doubles
px_origIntersec = sorted( px_origIntersec )
px_origIntersecOrganized = [[px_origIntersec[0]]] # End result is [[(i, j), (i, j+1), ...], [(i+1, k), (i+1, k+1), ...], ...]
list_index = 0
for i in range(1,len(px_origIntersec)):
    curr_tup = px_origIntersec[i]
    prev_tup = px_origIntersec[i-1]
    if( curr_tup[0] == prev_tup[0] ):
        px_origIntersecOrganized[list_index].append(curr_tup)    
    else:
        list_index += 1
        px_origIntersecOrganized.append([curr_tup])

dict_pxAreaInside = {} # Pixels fully inside of the circle 
for lst in px_origIntersecOrganized:
    col  = lst[0][0]
    rows = [ tup[1] for tup in lst ]
    next_colIfPos = -1
    for step in range( min(rows), max(rows) + 1 ):
        if( step in rows ):
            if( next_colIfPos >=0 ):
                next_colIfPos = -1
                continue
            else:
                pass
        else: 
            next_colIfPos += 1
            px_index = get_pixelIndex( ( col, step ), N_px )
            key = str(px_index)
            dict_pxAreaInside[key] = 1. # Area is 1 a.u.²

# Calculating area weights ---
dict_pxAreaFrac = {}
for key, val in dict_pxIntersec.items():
    px_index = int(key)
    px_orig = get_pixelOrigin( px_index, N_px )
    assert len(val)<=4, "ERROR in classifying intersections | Found too many intersections in pixel " + key
    if( len(val)==2 ):  # In most cases, there are two intersections.
        area = get_areaSegment( px_orig, val[0], val[1], R, x0, y0 )
    elif( len(val)==3 ): # In a rare case, you may be dead-on grazing the pixel boundary at a single point. Just ignore this point.
        x_insec = [ i[0] for i in val ]
        y_insec = [ i[1] for i in val ]
        for i in range(len(val)):
            index1 = int(i/2) # Just looping over (0,1), (0,2), (1,2)
            index2 = int((i+1)/2)+1
            if( type(x_insec[index1]) == type(x_insec[index2]) ): 
                area = get_areaSegment( px_orig, val[index1], val[index2], R, x0, y0 )
            else:
                pass
    elif( len(val)==4 ): # Experience has found that four intersections is possible. 
        x_insec = [ i[0] for i in val ]
        y_insec = [ i[1] for i in val ]
        yes_sameIntersecX = ( len(x_insec)!=len(set(x_insec)) ) # Only one of these two
        yes_sameIntersecY = ( len(y_insec)!=len(set(y_insec)) ) # should be true
        assert yes_sameIntersecX != yes_sameIntersecY, "ERROR in classifying intersections | Impossible case encountered for pixel " + key
        sort_along = int(yes_sameIntersecX) 
        val_ordered = sorted( val, key=lambda elem:elem[ sort_along ] ) # sort along x/y if there are two intersections on the same pixel row/column
        area = 0
        area += get_areaSegment( px_orig, val_ordered[0], val_ordered[1], R, x0, y0 )
        area += get_areaSegment( px_orig, val_ordered[2], val_ordered[3], R, x0, y0 )
        area += int(yes_sameIntersecY) * np.abs( val_ordered[2][0] - val_ordered[1][0] ) + int(yes_sameIntersecX) * np.abs( val_ordered[2][1] - val_ordered[1][1] ) 
    else:
        print( "ERROR in classifying intersections - Expecting 2 or 4 | Pixel " + key + " has " + str(len(val)) + " intersections." )
        exit()
    assert(area <= 1.), "ERROR in calculating fractional area | Area is larger than 1 for pixel " + key
    dict_pxAreaFrac[key] = area

for k,v in dict_pxAreaFrac.items():
    px_orig=get_pixelOrigin(int(k), N_px)
    # MC = Area_enclosedPixelMC( px_orig, R, x0, y0, 2000000 )
    # rel = (1-np.abs( MC-v )/ v)*100
    print( k, px_orig, v)#, MC, rel )

t1 = time.time()
t_algo = t1 -t0

# Check ---
t0 = time.time()
Ans = np.pi * R**2.
t1 = time.time()
t_anal = t1 -t0

t0 = time.time()
total_sum = 0
total_MC  = 0
N_MC = 1000000
for d in dict_pxAreaInside, dict_pxAreaFrac:
    for k,v in d.items():
        total_sum += v
        # px_orig = get_pixelOrigin( int(k), N_px )
        # total_MC += Area_enclosedPixelMC( px_orig, R, x0, y0, N_MC )

t1 = time.time()
t_MC = t1 -t0

print( "The analytical closed-form answer is: ", Ans, " and took ", t_anal, " sec to calculate" )
print( "The analytical algorithmic answer is: ", total_sum, " and took ", t_algo, " sec to calculate" )
# print( "The numerical  Monte-Carlo answer is: ", total_MC, " and took ", t_MC, " sec to calculate" )
# =================================================================================================

# Plotting ========================================================================================
fig, ax = plt.subplots()

N_plt = 1000 + 1
x     = np.linspace(grid_min, grid_max, num=N_plt, endpoint=True)

# Pixel grid ---
[ ax.vlines( x=i, ymin=grid_min, ymax=grid_max, color="gray", ls="--" ) for i in grid_sq ]
[ ax.hlines( y=i, xmin=grid_min, xmax=grid_max, color="gray", ls="--" ) for i in grid_sq ]

# ROI highlight ---
[ ax.vlines( x=ROI_x[i], ymin=ROI_y[0], ymax=ROI_y[-1], color="k" ) for i in range(len(ROI_x)) ]
[ ax.hlines( y=ROI_y[i], xmin=ROI_x[0], xmax=ROI_x[-1], color="k" ) for i in range(len(ROI_y)) ]

# Circle ---
y_pos, y_neg = circ( x, R, x0, y0 )
ax.plot(x, y_pos, color="tab:blue")
ax.plot(x, y_neg, color="tab:blue")

# Intersections between grid and circle ---
for i in range(len(intersec)):
    ax.plot( intersec[i][0], intersec[i][1], "x", color="tab:orange" )

# Line ---
y_L = Line( x, chi, x0, y0 )
ax.plot( x, y_L, color="tab:green" )
ax.plot( x0, y0, "x", color="tab:green" )

# Intersections between grid and line ---
for i in grid_sq:
    ax.plot( i, intersec_LineY( chi, x0, y0, i ), "^", color="tab:orange" )
    ax.plot( intersec_LineX( chi, x0, y0, i ), i, "^", color="tab:orange" )

# Example pixel highlight ---
px_origX = 5 
px_origY = 3  
ax.plot( px_origX, px_origY, "bo" )
ax.vlines( x=px_origX, ymin=px_origY, ymax=px_origY + 1, color="red" ) 
ax.vlines( x=px_origX + 1, ymin=px_origY, ymax=px_origY + 1, color="red" ) 
ax.hlines( y=px_origY, xmin=px_origX, xmax=px_origX + 1, color="red" ) 
ax.hlines( y=px_origY + 1, xmin=px_origX, xmax=px_origX + 1, color="red" ) 

# Pixel index numbering ---
fmt_len = 1 + int( math.log10(N_px*N_px) )
fmt_str = "{:0" + str(fmt_len) + "d}"


plt.show()

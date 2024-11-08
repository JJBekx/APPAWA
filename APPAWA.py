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

# TODO: Fill color AziLine is not consistently left (blue) and right (red)
# BUG: Analytical area always does a full circle - wrong if partially outside of grid

# Primordial classes that need no building blocks, only input parameters ==========
class PixelGrid():
    # -----------------------------------------------------------------------------
    # This class defines the square pixel grid and contains all of its properties. |
    # NOTE: Though px_len [μm] is provided, APPAWA's end results are fractional    |
    #       weights (equivalent essentially to px_len=1).                          |
    #------------------------------------------------------------------------------
    def __init__( self, N_px, px_len=75., **kwargs ): # ( self, int, float )
        super().__init__(**kwargs)
        self.N_px   = N_px   # Defines a square (N_px x N_px)-grid
        self.px_len = px_len # Defines the pixel width (=height) in μm - Default=Eiger2 # NOTE: verify with DanMAX

        grid_min = 0
        grid_max = N_px # Normalized to (1 x 1) a.u.² pixels. 
        self.grid_sq = range( grid_min, grid_max+1 ) # endpoint inclusive = N_px+1 points
        # Chose not to use a np.linspace, which uses np.int64 (=slow), instead of native int (=fast)

    @classmethod
    def get_pixelIndex( cls, px_orig, N_px ): # ( cls, (int, int), int )
        # ------------------------------------------------------------
        # Provides the counting index of a given pixel defined by its |
        # origin (lower-left corner) location, given as (x,y).        |
        # Example for a 3x3 pixel grid:                               |
        #                             ___________                     |
        #     e.g.: "0" = (0,0)      |_6_|_7_|_8_|                    |
        #           "5" = (2,1)      |_3_|_4_|_5_|                    |
        #           "7" = (1,2)      |_0_|_1_|_2_|                    |
        #                                                             |
        # ------------------------------------------------------------
        x_px, y_px = px_orig
        return N_px * y_px + x_px

    @classmethod
    def get_pixelOrigin( cls, px_index, N_px ): # ( cls, int, int )
        # ----------------------------------------
        # Provides the inverse of get_pixelIndex. |
        # ----------------------------------------
        return ( int( px_index%N_px ), int( px_index/N_px ) )

    @classmethod
    def is_in_grid( cls, px, N_px ): # (cls, (int, int), int)
        px_x, px_y = px
        val = ( px_x >= 0 and px_x <= N_px ) and ( px_y >= 0 and px_y <= N_px )
        return val

class Circle():
    # -----------------------------------------------------------------------------------
    # This class defines one of the circles of which diffraction rings are comprised of. |
    # Since APPAWA returns area fraction weights, it is assumed that R, x0, and y0 are   |
    # given in units of the px_len.                                                      |
    # The circle origin (x0,y0) is to be given with respect to the pixel-grid origin.    |
    #------------------------------------------------------------------------------------
    def __init__(self, R, x0, y0, **kwargs ): # ( self, float, float, float )
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)

        if( np.abs( chi ) > 2.*math.pi ):
            chi = chi * ( math.pi / 180. )
            print( "WARNING in creation of object instance AziLine \n" +
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
        yes_verticalLine   = ( math.isclose(chi,math.pi/2.) or math.isclose(chi,3.*math.pi/2.) )
        if( yes_verticalLine ): 
            return np.nan
        else:
            return y0 + ( x - x0 ) * math.tan( chi ) # See "Hesse normal form" for explanation

# Classes that build on primordial classes ==========
# TODO: init with multiple inheritance only calls init on the first parent
#       This is fixed by adding super().__init__(**kwargs) in the init of all parent classes
#       Better to create instances in the init of the child class instead? 

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
            # y_circ = CircleOnGrid.get_circ(x_rand[i], R, x0, y0)
            if( y_rand[i] >= y0 ):
                # yes_isNan = ( np.isnan( y_circ[0] ) )
                # y_U = N_px * int( yes_isNan ) + y_circ[0] * int(1 - yes_isNan)
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
    def get_areaSegment( cls, px_orig, intersec1, intersec2, R, x0, y0 ): # ( cls, (int, int), (float, float), (float, float), float, float, float )
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
        eps        = 1. - 5.e-16
        ROI_ixMin  = max( int( (x0 - R) ), 0 )
        ROI_ixMax  = min( int( (x0 + R) + eps ), N_px) # NOTE: Could be prone to bugs., but a tight ROI square is absolutely necessary
        ROI_x      = range( ROI_ixMin, ROI_ixMax+1 ) 
        self.ROI_x = ROI_x
        ROI_jyMin  = max( int( (y0 - R) ), 0 )
        ROI_jyMax  = min( int( (y0 + R) + eps ), N_px)
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

                if( y < 0 or y > N_px ): # Skip intersections outside of pixel grid
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
                    if( px_origX < 0 or px_origX >= N_px ): continue # Don't add "pixels" outside the grid
                    px_orig = ( px_origX, px_origY )
                    key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                    px_intersections.setdefault(key, []).append( intersection )
            
        for j in ROI_y: # Gathering all intersections with rows
            x_R, x_L = Circle.get_circ( x=j, R=R, x0=y0, y0=x0 ) # Can use same formula if you switch x <-> y.

            if( math.isclose( x_R, x_L ) ): # The pixel grid hits the circle exaclty when x_R=x_L
                continue                    # Ignore this, it was caught by y_U and y_D

            for x in x_L, x_R:
                if( np.isnan(x) ): # Skip NaNs
                    continue

                if( x < 0 or x > N_px ): # Skip intersections outside of pixel grid
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
                    if( px_origY < 0 or px_origY >= N_px ) : continue # Don't add "pixels" outside the grid
                    px_orig = ( px_origX, px_origY )
                    key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                    px_intersections.setdefault(key, []).append( intersection )

        self.px_intersections = px_intersections

        # Collecting areas ---
        px_enclosedArea = {} # Frac. pixel area enclosed in circle

        # Pixels in full interior of circle 
        keys = list( px_intersections.keys() )
        for i in range( ROI_ixMax + 1 ):
            for j in range( ROI_jyMax + 1 ):
                px_x     = int(x0) - (int(ROI_ixMax/2) + 1) + i
                px_y     = int(y0) - (int(ROI_jyMax/2) + 1) + j
                px       = (px_x, px_y)
                px_index = PixelGrid.get_pixelIndex( px, N_px )
                yes_alreadyCovered = px_index in keys
                if( yes_alreadyCovered ): continue
                yes_inGrid = PixelGrid.is_in_grid( px, N_px )
                dist = ( (x0-px_x)**2. + (y0-px_y)**2. )**0.5
                yes_within = dist < R
                if( yes_inGrid and yes_within ):
                    px_enclosedArea[px_index] = 1. # Area is 1 a.u.²
                else:
                    continue        

        # Fractional area weights 
        for key, val in px_intersections.items():
            px_orig = PixelGrid.get_pixelOrigin( key, N_px )

            if( len(val) <=1 or len(val) > 4 ): # Number of intersections should be 2, 3 or 4
                raise AssertionError( "ERROR in classifying Circle intersections | Pixel " + str(key) + " has " + str(len(val)) + " intersections." )

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
                    raise AssertionError( "ERROR in classifying Circle intersections | Impossible case encountered for pixel " + str(key) )
                sort_along = int(yes_sameIntersecX) 
                val_ordered = sorted( val, key=lambda elem:elem[ sort_along ] ) # sort along x/y if there are two intersections on the same pixel row/column
                area = 0
                area += CircleOnGrid.get_areaSegment( px_orig, val_ordered[0], val_ordered[1], R, x0, y0 )
                area += CircleOnGrid.get_areaSegment( px_orig, val_ordered[2], val_ordered[3], R, x0, y0 )
                area += int(yes_sameIntersecY) * np.abs( val_ordered[2][0] - val_ordered[1][0] ) + int(yes_sameIntersecX) * np.abs( val_ordered[2][1] - val_ordered[1][1] ) 
            else:
                raise AssertionError( "ERROR in classifying Circle intersections | Pixel " + str(key) + " has " + str(len(val)) + " intersections." )

            if (area > 1.):
                raise AssertionError( "ERROR in calculating Circle fractional area | Area is larger than 1 for pixel " + str(key) )

            px_enclosedArea[key] = area
        
        self.px_enclosedArea = px_enclosedArea

class AziLineOnGrid(PixelGrid, AziLine): 

    @classmethod 
    def get_areaSegment( cls, px_orig, intersec1, intersec2 ): # ( cls, (int, int), (float, float), (float, float) )
        # ----------------------------------------------------------------
        # Calculates the area to the left and to the right of the AziLine |
        # in the considered pixel. For vertical cut, Up = L               |
        #                                _____ /_                         |
        #                               |     /  |                        |
        #                               | L  /   |                        |
        #                               |   /  R |                        |
        #                                --/-----                         |
        # ----------------------------------------------------------------
        px_x, px_y = px_orig
        insec1, insec2 = sorted( [ intersec1, intersec2 ], key=lambda elem:elem[0] ) # sort along x-coordinate
        x1, y1 = insec1
        x2, y2 = insec2

        tria = (x2 - x1) * ( max(y1,y2) - min(y1,y2) ) / 2.

        yes_intersecTwoRows = ( type(y1)==int and type(y2)==int and max(y1,y2)-min(y1,y2)==1 )
        yes_intersecTwoCols = ( type(x1)==int and type(x2)==int and x2-x1==1 )
        if(yes_intersecTwoRows):
            block = x1 - px_x
        elif(yes_intersecTwoCols):
            block = min(y1,y2) - px_y 
        else:
            block = 0.

        area       = block + tria
        area_compl = 1. - area
        if( x2==px_x+1 ):
            if( yes_intersecTwoCols and y1 > y2 ):
                Left, Right = area, area_compl
            else:
                Left, Right = area_compl, area
        else:
            Left, Right = area, area_compl

        return Left, Right 

    def __init__( self, N_px, chi, x0, y0 ): 
        super().__init__( N_px=N_px, chi=chi, x0=x0, y0=y0 ) # Initialize parent attributes (does not make instances)

        yes_ascendingLine  = (chi >= 0. and chi < math.pi/2.) or (chi >= math.pi and chi < 3.*math.pi/2.) # no chi < 0
        yes_descendingLine = not yes_ascendingLine # BUG this is chi, not AziLine.chi. Error if someone fills in 183 deg
        # yes_horizontalLine = math.isclose( np.abs(chi%math.pi), 0. )
        yes_horizontalLine = ( math.isclose(chi,0.) or math.isclose(chi,math.pi) )
        yes_verticalLine   = ( math.isclose(chi,math.pi/2.) or math.isclose(chi,3.*math.pi/2.) )

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

            yes_intersecWithRow    = ( math.isclose( y, int(np.rint(y)) ) ) # The intersection is at a pixel row - possible four-way intersection
            yes_intersecGridBottom = ( math.isclose( y, grid_sq[0] ) )      # The intersection hits the pixel grid bottom
            yes_intersecGridTop    = ( math.isclose( y, grid_sq[-1] ) )     # The intersection hits the pixel grid top
            yes_intersecFourWay    = yes_intersecWithRow and not ( yes_intersecGridBottom or yes_intersecGridTop )

            for col in 0,1: # Adding pixels right (i) and left (i-1) of the y intersection in one loop
                if( yes_intersecFourWay ): # Hit a full-on four-way intersection - Add diagonally adjacent pixels...
                    px_origX = i + ( int(-col)*int(yes_ascendingLine) + int(col-1)*int(yes_descendingLine) )
                    px_origY = int(np.rint(y)) + int(-col) * int(1-yes_horizontalLine) #..unless the line is horizontal, then add pixels left and right.
                else:
                    yes_needsOriginFix = yes_intersecGridTop # Don't count pixel outside the grid
                    px_origX = i - col
                    px_origY = int(y) - int(yes_needsOriginFix)
                if( px_origX < 0 or px_origX >= N_px ): continue # Don't add "pixels" outside the grid
                px_orig = ( px_origX, px_origY )
                key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                px_intersections.setdefault(key, []).append( intersection )
            
        for j in grid_sq: # Gathering all intersections with rows
            if( math.isclose(abs(chi), math.pi/4.) ): continue # catch bordercase to prevent third intersection
            x = AziLine.get_line( x=j, chi=math.pi/2.-chi, x0=y0, y0=x0 ) # Can use same formula if you switch x <-> y.
            # Switch x <-> equivalent to chi -> pi/2 - chi, x0 -> y0, y0 -> x0

            if( np.isnan(x) ): # Skip NaNs.
                continue

            if( x < 0 or x > N_px ): # Skip intersections outside of pixel grid
                continue

            yes_intersecWithCol   = ( math.isclose( x, int(np.rint(x)) ) ) # The intersection is at a pixel col - possible four-way intersection
            yes_intersecGridLeft  = ( math.isclose( x, grid_sq[0] ) )      # The intersection hits the grid on the left
            yes_intersecGridRight = ( math.isclose( x, grid_sq[-1] ) )     # The intersection hits the grid on the right
            yes_intersecFourWay   = yes_intersecWithCol and not ( yes_intersecGridLeft or yes_intersecGridRight )

            if( yes_intersecFourWay and not yes_verticalLine ): # Caught the four-way intersections in the y-loop above (if line isn't vertical)
                continue
            
            intersection = (x, j)

            yes_needsOriginFix = yes_intersecGridRight # Don't count pixel to the right of the circle
            for row in 0,1: # Adding the pixels above (j) and below (j-1) of the x intersection in one loop
                px_origX = int(np.rint(x)) * int(yes_intersecWithCol) + int(x) * int(1-yes_intersecWithCol) - int(yes_needsOriginFix)
                px_origY = j - row
                if( px_origY < 0 or px_origY >= N_px ) : continue # Don't add "pixels" outside the grid
                px_orig = ( px_origX, px_origY )
                key = PixelGrid.get_pixelIndex( px_orig=px_orig, N_px=N_px )
                px_intersections.setdefault(key, []).append( intersection )

        self.px_intersections = px_intersections

        # Collecting areas ---
        px_enclosedArea = {} # Frac. area left and right of the AziLine
        # All pixels not in AziLineOnGrid.px_enclosedArea have an area of 1
        # Different than in CircleOnGrid.px_enclosedArea

        # Fractional area weights 
        for key, val in px_intersections.items():
            px_orig = PixelGrid.get_pixelOrigin( key, N_px )

            if( len(val) !=2 ): # Number of intersections should be 2
                raise AssertionError( "ERROR in classifying AziLine intersections | Pixel " + str(key) + " has " + str(len(val)) + " intersections." )

            area_L, area_R = AziLineOnGrid.get_areaSegment( px_orig, val[0], val[1] )
            
            if (area_L + area_R > 1.):
                raise AssertionError( "ERROR in calculating fractional area | Area is larger than 1 for pixel " + str(key) )

            px_enclosedArea[key] = area_L, area_R 
        self.px_enclosedArea = px_enclosedArea

# Functions designed for development checks ==========
def check_CoGIntersections( N_px, R, x0, y0, N_plt=1000+1 ): # ( int, float, float, float, int )
    
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
    ax.plot(x0, y0, "bo")
    y_upper = []; y_lower = []
    y_forFillUp = []; y_forFillDown = []
    for i in x_plt:
        y_U, y_D = CoG.get_circ( i, R, x0, y0 ) 
        yes_isNanU = np.isnan( y_U )
        yes_isNanD = np.isnan( y_D )
        y_forFillUp.append( min(N_px, y_U) * int( 1-yes_isNanU ) + y_U * int(yes_isNanU) )
        y_forFillDown.append( max(0, y_D) * int( 1-yes_isNanD ) + y_D * int(yes_isNanD) )
        if( y_U < 0 or y_U > N_px ): y_U = np.nan
        if( y_D < 0 or y_D > N_px ): y_D = np.nan
        y_upper.append(y_U)
        y_lower.append(y_D)
    ax.plot(x_plt, y_upper, color="tab:blue")
    ax.plot(x_plt, y_lower, color="tab:blue")
    ax.fill_between( x=x_plt, y1=y_forFillDown, y2=y_forFillUp, color="red", alpha=0.5 )

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

def check_ALoGIntersections( N_px, chi, x0, y0, N_plt=1000+1 ): # ( int, float, float, float, int )
    
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
    ax.hlines( y=y0, xmin=0, xmax=N_px, color="k", ls="-" )

    # Line 
    y_line = [ ALoG.get_line( i, chi, x0, y0 ) for i in x_plt ]
    ax.plot(x_plt, y_line, color="tab:blue")
    if( math.isclose( chi, math.pi/2. ) or math.isclose( chi, 3.*math.pi/2. ) ):
        ax.vlines( x=x0, ymin=0, ymax=N_px, color="tab:blue" )

    yes_verticalLine   = ( math.isclose(chi,math.pi/2.) or math.isclose(chi,3.*math.pi/2.) )
    if( yes_verticalLine ):
        ax.fill_betweenx(y=range(0, N_px+1), x1=0, x2=x0, color="blue", alpha=0.5 )
        ax.fill_betweenx(y=range(0, N_px+1), x1=x0, x2=N_px, color="red", alpha=0.5 )
    else:
        y_L = []; y_R = []
        for i in x_plt:
            y_line = ALoG.get_line( i, chi, x0, y0 )
            y_L.append(max(0., y_line))
            y_R.append(min(N_px, y_line))
        ax.fill_between(x=x_plt, y1=y_L, y2=np.ones(len(x_plt))*N_px, color="blue", alpha=0.5 )
        ax.fill_between(x=x_plt, y1=np.zeros(len(x_plt)), y2=y_R, color="red", alpha=0.5 )

    # Intersections between grid and circle 
    for i in range(len(px_intersections)):
        ax.plot( px_intersections[i][0], px_intersections[i][1], "x", color="tab:orange" )

    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    plt.show()
    return 

def check_ALoGAreas( N_px, chi, x0, y0 ): # ( int, float, float, float )
    ALoG = AziLineOnGrid( N_px, chi, x0, y0 )
    for k,v in ALoG.px_enclosedArea.items():
        px_origin = PixelGrid.get_pixelOrigin( k, N_px )
        print( "Pixel ", k, " at loc", px_origin, " is cut into L & R areas", v, "; Total=", sum(v) )
    return

# Testing Space ============

N_px = 10
x0 = 5.
y0 = 5.

# Circle on Grid ---
R = 2.
# check_CoGIntersections( N_px=N_px, R=R, x0=x0, y0=y0 )
# check_CoGAreas(N_px=N_px, R=R, x0=x0, y0=y0, yes_MC=True, N_MC=1000000, yes_verbose=True)

# AziLine on Grid ---
chi = math.pi * (5/6)
check_ALoGIntersections( N_px=N_px, chi=chi, x0=x0, y0=y0 )
check_ALoGAreas( N_px=N_px, chi=chi, x0=x0, y0=y0 )

# ===========================
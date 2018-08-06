
# Python Functionality for Creating Walk-Enriched Sample and Interpolating Fixed-Effects Raster ------------------------


# Step 1: Geocode point of interest and generate service areas ---------------------------------------------------------

def service_area(address, title, already_created="no", write="no",
                 grocery_time=10, health_time=10, schools_time=10, transit_time=10):
    import arcgis

    user_name = 'aweinstock_dcdev'
    password = ']6wiA6gmabw!'
    my_gis = arcgis.gis.GIS("https://dcdev.maps.arcgis.com/", user_name, password)

    if already_created == "no":
        print("Reading point of interest...")
        poi = arcgis.geocoding.geocode(address=address, max_locations=1, as_featureset=True)

        print("Creating service area from maximum specified walk-time distance...")
        radius = max([grocery_time, health_time, schools_time, transit_time])
        sa = arcgis.network.analysis.generate_service_areas(facilities=poi,
                                                            break_values=str(radius),
                                                            break_units="Minutes",
                                                            travel_direction="Away From Facility",
                                                            travel_mode="Walking")
        sa_feature = sa[0]

        if write != "no":
            print("Writing service area polygons publicly to the GIS...")
            as_df = sa_feature.df
            sa_item = my_gis.content.import_data(df=as_df,
                                                 title=title,
                                                 tags="vector_test_walkability")
            sa_item.share(everyone=True)

        print("Complete. Service area generation successfully finished")
        interest_area = {'point': poi, 'service_area': sa_feature}
        return interest_area

    if already_created != "no":
        print("Obtaining feature layer for already-created service area polygons")
        search = my_gis.content.search(''.join(["title:", title]), item_type="Feature Layer")
        sa_item = search[0]
        service_area_lyr = sa_item.layers[0]

        print("Complete. Extraction of URL for service areas successfully finished")
        return service_area_lyr


# Step 2: Calculate and extract walk time distances to DUCs from each sample point -------------------------------------

def minute_extraction(service_area_title, address,
                      grocery_time=10, health_time=10, schools_time=10, transit_time=10,
                      grocery_weight=0.25, health_weight=0.25, schools_weight=0.25, transit_weight=0.25):
    import arcgis
    import math

    user_name = 'aweinstock_dcdev'
    password = ']6wiA6gmabw!'
    my_gis = arcgis.gis.GIS("https://dcdev.maps.arcgis.com/", user_name, password)
    grocery_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_Grocery/FeatureServer/0"
    health_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_Health/FeatureServer/0"
    schools_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_Schools/FeatureServer/0"
    transit_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_Transit/FeatureServer/0"

    print("Reading in the service area polygons...")
    sa = service_area(title=service_area_title, address=address, already_created="yes")
    servar = sa.query()

    print("Defining point of interest")
    poi = arcgis.geocoding.geocode(address=address, max_locations=1, as_featureset=True)

    print("Setting up framework for minutes extraction...")
    minutes_dict = {}
    indicators = ["grocery", "health", "schools", "transit"]
    specs = {"grocery": grocery_time, "health": health_time, "schools": schools_time, "transit": transit_time}
    weights = {"grocery": grocery_weight, "health": health_weight, "schools": schools_weight, "transit": transit_weight}
    urls = {"grocery": grocery_url, "health": health_url, "schools": schools_url, "transit": transit_url}
    point_display = {}

    print("Commencing minutes extractions to all indicators")
    for j in indicators:
        print("Working on " + j)
        if weights[j] != 0:
            lyr = arcgis.features.FeatureLayer(url=urls[j])
            feat = lyr.query(geometry_filter=arcgis.geometry.filters.contains(geometry=servar.features[0].geometry))
            if len(feat.features) != 0:
                odcm = arcgis.network.analysis.generate_origin_destination_cost_matrix(origins=poi,
                                                                                       destinations=feat,
                                                                                       travel_mode="Walking",
                                                                                       time_units="Minutes")
                dist_mat = odcm[1].df
                walktimes = list(dist_mat['Total_Time'])
                rounded = [math.ceil(w) for w in walktimes]
                minutes_dict[j] = [m for m in rounded if m <= specs[j]]

                dest = odcm[3].df
                dest.insert(1, 'time', walktimes)
                toplot = dest[dest['time'] <= specs[j]]
                if len(toplot) > 0:
                    toplot_feat = arcgis.features.FeatureSet.from_dataframe(df=toplot)
                    point_display[j] = toplot_feat

            else:
                minutes_dict[j] = []
        else:
            minutes_dict[j] = []

    print("Complete. Calculation and extraction of walk time distances to indicators successfully finished.")
    return [minutes_dict, point_display]


# Step 3: Call the function to obtain "area swing" function data -------------------------------------------------------

def swing_input(service_area_title, address):
    import pandas
    import arcgis

    user_name = 'aweinstock_dcdev'
    password = ']6wiA6gmabw!'
    my_gis = arcgis.gis.GIS("https://dcdev.maps.arcgis.com/", user_name, password)
    crashes_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_CrashesINC/FeatureServer/0"
    crime_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_Crime/FeatureServer/0"
    historic_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_Historic/FeatureServer/0"
    trees_url = "https://services.arcgis.com/bkrWlSKcjUDFDtgw/arcgis/rest/services/MAN_Trees/FeatureServer/0"

    print("Reading in the service area polygons...")
    sa = service_area(title=service_area_title, address=address, already_created="yes")
    servar = sa.query()

    print("Setting up framework for swing parameter input (SPI) data extraction...")
    urls = {"crashes": crashes_url, "crime": crime_url, "historic": historic_url, "trees": trees_url}
    per_df = pandas.DataFrame(data={'point_id': [1]})
    consid = ["crashes", "crime", "historic", "trees"]

    print("Commencing calculation of percent of each swing input occurring in the service area...")
    for i in consid:
        print("Working on " + i)
        char = arcgis.features.FeatureLayer(url=urls[i])
        n_total = len(char.query().features)
        num = char.query(geometry_filter=arcgis.geometry.filters.contains(geometry=servar.features[0].geometry),
                         return_count_only=True)
        p = num / n_total
        per_df[i] = p
        print("Data from " + i + " SPI file successfully saved...")

    print("Complete. Calculation and extraction of SPI percentages by service area successfully finished.")
    return per_df


# Step 4: Call all other the functions necessary to calculate a category walk score ------------------------------------

def minimax(x):
    mmn = []
    for val in x:
        mmn.append((val-min(x))/(max(x)-min(x)))
    return mmn


def k_est(fl=100, dn=1, bw=0.2):
    import numpy
    from scipy.optimize import fsolve
    l = fl
    n_c = dn
    p = bw
    if p < 0.5:
        # print("Solve for k by coercing the weight for point with closeness rank " + str(n_c + 1) + " to " + str(p))
        func = lambda k: (1 / (1 + numpy.exp(k * (n_c + 1 - n_c - 0.5))) - 1 / (1 + numpy.exp(k * (l - n_c - 0.5)))) / (p * (1 / (1 + numpy.exp(k * (1 - n_c - 0.5))) - 1 / (1 + numpy.exp(k * (l - n_c - 0.5))))) - 1
        k_init = 2 * numpy.log((1 - p) / p)
        k_sol = fsolve(func, k_init)
        k_sol = k_sol[0]
    elif p > 0.5 and n_c != 1:
        # print("Solve for k by coercing the weight for point with closeness rank " + str(n_c) + " to " + str(p))
        func = lambda k: (1 / (1 + numpy.exp(k * (n_c - n_c - 0.5))) - 1 / (1 + numpy.exp(k * (l - n_c - 0.5)))) / (p * (1 / (1 + numpy.exp(k * (1 - n_c - 0.5))) - 1 / (1 + numpy.exp(k * (l - n_c - 0.5))))) - 1
        k_init = 2 * numpy.log((1 - p) / p)
        k_sol = fsolve(func, k_init)
        k_sol = -k_sol[0]
    elif p > 0.5 and n_c == 1:
        # print("Cannot coerce normalization for dn = 1 case to a weight != 1. By default, the 2nd point will be coerced to a weight = " + str(1 - p) + ". Please re-check the input parameters if this is an issue")
        p = 1 - p
        func = lambda k: (1 / (1 + numpy.exp(k * (n_c + 1 - n_c - 0.5))) - 1 / (1 + numpy.exp(k * (l - n_c - 0.5)))) / (p * (1 / (1 + numpy.exp(k * (1 - n_c - 0.5))) - 1 / (1 + numpy.exp(k * (l - n_c - 0.5))))) - 1
        k_init = 2 * numpy.log((1 - p) / p)
        k_sol = fsolve(func, k_init)
        k_sol = k_sol[0]
    # print("Steepness parameter k = " + str(round(k_sol, 4)))
    return k_sol


def close(fun_len=100, desired_number=1, boundary_weight=0.2, steepness=None):
    import numpy
    import matplotlib.pyplot as plot
    import pandas
    l = numpy.arange(1, fun_len + 1, 1)
    if steepness:
        steep = steepness
        # print("Steepness parameter k = " + str(steep))
    else:
        steep = k_est(fl=fun_len, dn=desired_number, bw=boundary_weight)
    w = [1/(1 + numpy.exp(steep * (i - desired_number - 0.5))) for i in l]
    w = minimax(w)
    # plot.plot(l, w)
    # plot.axis([1, 10, 0, 1])
    # plot.xlabel("Point Rank (by Time to Origin)")
    # plot.ylabel("Weight")
    # plot.title("Decay Function for Closeness Weighting")
    # plot.show()
    close_df = pandas.DataFrame({"clrank": l, "close_weight": w})
    return close_df


def area_swing(crashes=0, crime=0, historic=0, trees=0):
    swing = 0 + 100 * (- crashes - crime + historic + trees)
    # print("Area Swing Parameter = ", + str(round(swing, 7)))
    return swing


def dist(walkable_distance=10):
    import matplotlib.pyplot as plot
    import pandas
    d = list(range(1, walkable_distance+1))
    w = [1-i for i in d]
    wm = minimax(w)
    wmu = [i * 0.5 + 0.5 for i in wm]
    # plot.plot(d, w)
    # plot.axis([1, max(d), 0, 1])
    # plot.xlabel("Distance (in Minutes)")
    # plot.ylabel("Weight")
    # plot.title("Decay Function for Distance Weighting")
    # plot.show()
    dist_df = pandas.DataFrame({"distance": d, "dist_weight": wmu})
    return dist_df


def weighter(minutes, walkable_distance=10, fun_len=100, desired_number=1, boundary_weight=0.2, steepness=None):
    if len(minutes) == 0:
        walk = 0
        # print("Category Walk Score = " + str(walk) + " --- no points in a walkable distance of the origin")
    else:
        import numpy
        import pandas
        in_area = pandas.DataFrame({"num": numpy.arange(1, len(minutes) + 1, 1), "time": sorted(minutes)})
        dist_decay = dist(walkable_distance=walkable_distance)
        close_decay = close(fun_len=fun_len, desired_number=desired_number, boundary_weight=boundary_weight, steepness=steepness)
        dweight = []
        for i in in_area.time:
            dweight.append(dist_decay[dist_decay.distance == i].dist_weight.values[0])
        cweight = close_decay.head(max(in_area.num))['close_weight']
        # weight = pandas.DataFrame({"pt_dist": in_area.time, "pt_close": in_area.num, "dist_weight": dweight, "close_weight": cweight})
        # print("Table of Points, Distances, and Weights:")
        # print(weight)
        walk_first = numpy.dot(dweight, cweight)
        normalizer = numpy.sum(close_decay['close_weight'][0:desired_number])
        if walk_first/normalizer > 1:
            walk = 1
        else:
            walk = walk_first/normalizer
        # print("Category Walk Score = " + str(round(walk, 7)))
    return walk


# Step 5: Call the function to calculate walkability scores at each point ----------------------------------------------

def walkability(minutes_dict, swing_inputs,
                grocery_weight=0.25, health_weight=0.25, schools_weight=0.25, transit_weight=0.25,
                grocery_time=10, health_time=10, schools_time=10, transit_time=10,
                grocery_number=1, health_number=1, schools_number=1, transit_number=1,
                bound=0.2):

    print("Setting up for walk score calculations")
    inputs = {"grocery": {}, "health": {}, "schools": {}, "transit": {}}
    inputs["grocery"] = {"weight": grocery_weight, "time": grocery_time, "number": grocery_number}
    inputs["health"] = {"weight": health_weight, "time": health_time, "number": health_number}
    inputs["schools"] = {"weight": schools_weight, "time": schools_time, "number": schools_number}
    inputs["transit"] = {"weight": transit_weight, "time": transit_time, "number": transit_number}

    print("Calculating walk score...")
    cs = []
    for j in inputs.keys():
        w = inputs[j]["weight"] * 100
        cs.append(w * weighter(minutes=minutes_dict[j],
                               walkable_distance=inputs[j]["time"],
                               desired_number=inputs[j]["number"],
                               boundary_weight=bound))
    swing = area_swing(crashes=swing_inputs[swing_inputs['point_id'] == 1]["crashes"],
                       crime=swing_inputs[swing_inputs['point_id'] == 1]["crime"],
                       historic=swing_inputs[swing_inputs['point_id'] == 1]["historic"],
                       trees=swing_inputs[swing_inputs['point_id'] == 1]["trees"])
    total_walk = list(sum(cs) + swing)[0]
    if total_walk < 0:
        final = 0
    else:
        final = total_walk
    walk_score = final

    print("Complete. Calculation of walk scores successfully finished.")
    return walk_score


# Step 6: Call the function to execute the process, and write walk score enriched file ---------------------------------

def walk_update(city, address, service_area_title, web_map_title,
                grocery_weight=0.25, health_weight=0.25, schools_weight=0.25, transit_weight=0.25,
                grocery_time=10, health_time=10, schools_time=10, transit_time=10,
                grocery_number=1, health_number=1, schools_number=1, transit_number=1,
                bound=0.2):

    print("--- Walkability Analysis for " + city + " ---")

    print("--- Step 1: Service Area Generation ---")
    sa = service_area(address=address, title=service_area_title, already_created="no", write="yes",
                      grocery_time=grocery_time, health_time=health_time, schools_time=schools_time, transit_time=transit_time)

    print("--- Step 2: Minute Calculation & Extraction ---")
    minutes = minute_extraction(service_area_title=service_area_title, address=address,
                                grocery_time=grocery_time, health_time=health_time, schools_time=schools_time, transit_time=transit_time,
                                grocery_weight=grocery_weight, health_weight=health_weight, schools_weight=schools_weight, transit_weight=transit_weight)

    print("--- Step 3: Swing Parameter Input (SPI) Collection ---")
    swing = swing_input(service_area_title=service_area_title, address=address)

    print("--- Step 4: Walkability Calculation ---")
    walk_score_value = walkability(minutes_dict=minutes[0], swing_inputs=swing,
                                   grocery_weight=grocery_weight, health_weight=health_weight, schools_weight=schools_weight, transit_weight=transit_weight,
                                   grocery_time=grocery_time, health_time=health_time, schools_time=schools_time, transit_time=transit_time,
                                   grocery_number=grocery_number, health_number=health_number, schools_number=schools_number, transit_number=transit_number,
                                   bound=bound)

    print("--- Step 5: Plotting Output ---")
    import arcgis
    user_name = 'aweinstock_dcdev'
    password = ']6wiA6gmabw!'
    my_gis = arcgis.gis.GIS("https://dcdev.maps.arcgis.com/", user_name, password)

    print("Defining potential map symbols...")
    symbols = {"grocery": {"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"http://static.arcgis.com/images/Symbols/PeoplePlaces/Shopping.png","contentType":"image/png","width":24,"height":24},
               "health": {"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"http://static.arcgis.com/images/Symbols/SafetyHealth/Ambulance.png","contentType":"image/png","width":24,"height":24},
               "schools": {"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"http://static.arcgis.com/images/Symbols/PeoplePlaces/School.png","contentType":"image/png","width":24,"height":24},
               "transit": {"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"http://static.arcgis.com/images/Symbols/Transportation/esriDefaultMarker_197_Yellow.png","contentType":"image/png","width":24,"height":24}}


    print("Creating the user's map...")
    map = my_gis.map(location=arcgis.geocoding.geocode(address=address, max_locations=1))
    map.draw(sa['point'])
    map.draw(sa['service_area'])
    for k in minutes[1]:
        symbol = symbols[k]
        map.draw(minutes[1][k], symbol=symbol)

    print("Saving the created map...")
    wm = map.save(item_properties={"title": web_map_title, "snippet": "a walkability vector side webmap", "tags": "vector_walkability"})

    print("Got it!")

    print("--- Process complete! ---")
    return [walk_score_value, wm]


# Step 7: Run it! ------------------------------------------------------------------------------------------------------

# An Example:
nyc_walkscore = walk_update(city="Manhattan",
                            address="350 5th Ave, New York, NY 10118",
                            service_area_title="Vector_Side_Test_SA",
                            web_map_title="Empire_State_Walkability")


# --- IN TOTALITY, HERE'S WHAT WE HAVE TO DO: --------------------------------------------------------------------------

# run "walk_update" to get walkability score



# Flask App for The Airbnb Listings in Beijing Analysis Project

This project aimed to help the host set the price of their new listing. There are three parts of analysis in this project. 

1. whole city analysis

Listing distribution, room type, availability of the listings, host type of the whole city and each district, price of the listings, income of the host are analyzed and shown in this part.
Host can get an overview of the situation of the whole city.

![image](https://github.com/Hannah-Yu-0816/Airbnb_Listing_in_Beijing_Analaysis/blob/main/flask/static/img/distribution.png)

2. Each district analysis

Room type, host type, price, availability, and income information of certain district are shown in this part. If host has a listing in a certain district,
he/she can select the district from the home page and then click the "submit" button to get the information of the district.

3. Get your own advice

To give the host some advice on the price of their new listing, K-NN method is used to get the nearest k neighbors of the new listing. 
8 variables, including the information of the location, host and room, are used in trainging the model. 
For example, the host has a new listing, and he/she would like to set the price of the listing to rent.
His/Her listing is in Chaoyang district, the latitude and longitude are 41 and 116, room type is Entire home/apt.
He/she has another 2 listings so he/she belongs to Personal host.
The room can accommodate up to four people, and minimum 1 night should be booked.
This listing will be available 20 days per month.
After entering all these information and clicking the button of "Get your own advice", the top k most similar listings are retrived by K-NN method.
The basic information of the 5 listings are shown and the price information can be considered as the suggestion.

We suggest set the mean of the price of the similar listings as the price of the new listing, 
adjust the price between 1/4 quantile and 3/4 quantiles of these prices.
However, to get more accurate price suggesstion, we suggest the host review the "description" and "amenities" of these similar listings to compare and get some detail information.

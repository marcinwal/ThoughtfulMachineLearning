require 'matrix'

def distance(v1,v2)
  (v1-v2).magnitude
end

house_happines = {
  Vector[56,2] => "Happy",
  Vector[2,20] => "Not Happy",
  Vector[18,1] => "Happy",
  Vector[20,14] => "Not Happy",
  Vector[30,30] => "Happy",
  Vector[35,35] => "Happy"
}

house_1 = Vector[10,10]
house_2 = Vector[40,40]

def find_nearest(house,house_happines)
  house_happines.sort_by{|point,v|
    distance(house,point)
  }.first
end

def find_nearest_with_k(house,house_happines,k)
  house_happines.sort_by{|point,v|
    distance(point,house)
    }.first(k)
end

puts find_nearest(house_1,house_happines)
puts find_nearest(house_2,house_happines)

puts find_nearest_with_k(house_1,house_happines,3)
puts find_nearest_with_k(house_2,house_happines,3)




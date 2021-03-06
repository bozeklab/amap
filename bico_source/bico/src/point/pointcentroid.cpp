#include "../point/pointcentroid.h"
#include "../point/point.h"

#include <vector>

using namespace CluE;

PointCentroid* PointCentroid::clone() const
{
	return new PointCentroid(*this);
}

Point PointCentroid::generate(std::vector<Point*> const& source) const
{
    return Point(source);
}

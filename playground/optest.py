
# precedence:
# **
# * @ / // %
# + -
# << >>
# &
# ^
# |
# < <= > >= != ==
# and
# or

class Composable(type):
    def __gt__(cls, other):
        print("cls, other:", cls, other)
        meta = type(cls)
        print("meta:", meta)
        new_cls = meta(cls.__name__ + ">" + other.__name__, (cls, other), {})
        print("new_cls:", new_cls)
        out1 = cls.__dict__["out"]
        out2 = other.__dict__["out"]

        def new_out(self, ts):
            return out1(self, out2(self, ts))
        setattr(new_cls, "out", new_out)
        return new_cls


class Mod(metaclass=Composable):
    def out(self, ts):
        print("Not implemented yet")

    def __call__(self, ts):
        out = self.out(ts)
        return out


class Duplicator(Mod):
    def out(self, ts):
        return ts + ts


class Repeater(Mod):
    def out(self, ts):
        return [x for y in ts for x in [y,y]]


d = Duplicator()
print(d.out([1,2,3]))
r = Repeater()
print(r.out([1,2,3]))

Dupeater = Duplicator > Repeater

print("dupeater", Dupeater)
print("type dupeater", type(Dupeater))

dr = Dupeater()
print("type dr", type(dr))
print(dr.out([1,2,3]))



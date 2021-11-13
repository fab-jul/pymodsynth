
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

# Done:
# F(G()) -> F | G
# F(G(I())) -> F | G | I
# Next up:
# F(G(), H()) -> F | G & H             # & binds tighter than | and works like a comma.
# F(G(I()), H()) -> F | (G | I) & H    # must rely on brackets again? that makes everything pointless xD
#                                      # we could have several ops with the same semantics but different precedences
#                                      # and then use | or / depending on how tightly we want to bind
# F(G(I()), H()) -> F | G / I & H      # <- like this. but now & must have precedence between / and |...
# F(G(I()), H(J())) -> F | G / I & H / J        # this will all look terribly unclear.
# F(G(I(), H()) -> F | G | I & H


class Composable(type):
    def __or__(cls, other):
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

Dupeater = Duplicator | Repeater

print("dupeater", Dupeater)
print("type dupeater", type(Dupeater))

dr = Dupeater()
print("type dr", type(dr))
print(dr.out([1,2,3]))

Sickerator = Duplicator | Repeater | Duplicator
sick = Sickerator()
print(sick.out([1,2,3]))


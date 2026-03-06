#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Position {
    pub r: i8,
    pub c: i8,
}

impl Position {
    pub const fn add(&self, other: Self) -> Self {
        let new_r = self.r + other.r;
        let new_c = self.c + other.c;
        return Self { r: new_r, c: new_c };
    }
}

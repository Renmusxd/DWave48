use std::fmt::Debug;
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
pub struct Arena<T: Clone+Debug> {
    arena: Vec<T>,
    default: T,
    index: usize
}

impl<T: Clone+Debug> Arena<T> {
    pub fn new(default: T) -> Self {
        Self {
            arena: vec![],
            default,
            index: 0
        }
    }

    pub fn clear(&mut self) {
        self.index = 0
    }

    pub fn get_alloc(&mut self, size: usize) -> ArenaIndex {
        let index = self.index;
        let def_ref = &self.default;
        self.arena.resize_with(index+size, || def_ref.clone());
        let index = ArenaIndex {
            start: index,
            stop: index + size
        };
        self.index += size;
        index
    }
}

impl<T: Clone+Debug> Index<&ArenaIndex> for Arena<T> {
    type Output = [T];

    fn index(&self, index: &ArenaIndex) -> &Self::Output {
        &self.arena[index.start .. index.stop]
    }
}

impl<T: Clone+Debug> IndexMut<&ArenaIndex> for Arena<T> {
    fn index_mut(&mut self, index: &ArenaIndex) -> &mut Self::Output {
        &mut self.arena[index.start .. index.stop]
    }
}

pub struct ArenaIndex {
    start: usize,
    stop: usize
}

impl ArenaIndex {
    pub fn size(&self) -> usize {
        self.stop - self.start
    }
}
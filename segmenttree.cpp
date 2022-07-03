#include <vector>

class segtree {
    public:

    /* 이진 트리는 배열로 구현하는게 국룰. */
    std::vector<double> tree;
    int maxlen;

    segtree(int maxlen) {
        tree=std::vector<double>(maxlen*4+1, 0);
        this->maxlen=maxlen;
    }

    /* 트리의 탐색, 수정은 모두 재귀로 함
    사흘 전에 올린 게시물 참조
    start: 현재 노드가 가리키는 인덱스 구간의 시작점
    end: 현재 노드가 가리키는 구간의 끝점
    goal_idx: 수정하고자 하는 부분(인덱스)
    mid: (start+end) // 2
    왼쪽 자식은 tree 벡터의 tree_idx*2 번째 칸에 있으며, 구간 start~mid까지를 가리킴 
    오른쪽 자식은 tree 벡터의 tree_idx*2+1 번째 칸에 있으며, 구간 mid+1~end 까지를 가리킴 */
    void edit_tree(int start, int end, int goal_idx, int tree_idx, double val) {
        tree[tree_idx]+=val;
        if (start!=end) {
            int mid=(start+end)/2;
            if (goal_idx<=mid) {
                edit_tree(start, mid, goal_idx, tree_idx*2, val);
            }
            else edit_tree(mid+1, end, goal_idx, tree_idx*2+1, val);
        }
    }

    int get_idx(int start, int end, double goal, int tree_idx) {
        if (start==end) return start;
        int mid=(start+end)/2;
        if (tree[tree_idx*2]>=goal) return get_idx(start, mid, goal, tree_idx*2);
        return get_idx(mid+1, end, goal-tree[tree_idx*2], tree_idx*2+1);
    }
};

/* ctypes 모듈에서 클래스의 메소드를 그대로 불러올 순 없고
C식으로 객체의 포인터를 받아서 거기다가 뭘 하는 함수가 필요함 */

extern "C" {
    segtree* new_tree(int maxlen) {
        return new segtree(maxlen);
    }

    void tree_edit(void* ptr, int idx, double val) {
        segtree* self_tree=(segtree*)ptr;
        self_tree->edit_tree(1, self_tree->maxlen, idx, 1, val);
    }

    int tree_get_idx(void* ptr, double goal) {
        segtree* self_tree=(segtree*)ptr;
        return self_tree->get_idx(1, self_tree->maxlen, goal, 1);
    }

    double total(void* ptr) {
        segtree* self_tree=(segtree*)ptr;
        return self_tree->tree[1];
    }
}

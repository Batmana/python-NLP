# -*- coding: UTF-8 -*-


class Node(object):
    def __init__(self, value) -> None:
        self._children = {}
        self._value = value

    def _add_child(self, char, value, overwrite = False):
        child = self._children.get(char)
        if child is None:
            child = Node(value)
            self._children[char] = child
        elif overwrite:
            child._value = value

        return child


class Trie(Node):
    """
    树模型
    """
    def __init__(self) -> None:
        """
        构造函数
        """
        super().__init__(Node)

    def __contains__(self, key):
        """
        魔术方法，定义当使用成员测试运算符（in 或 not in）时的行为
        :param key:
        :return:
        """
        return self[key] is not None

    def __getitem__(self, key):
        """
        魔术方法，定义获取容器中指定元素的行为，相当于 self[key]
        :param item:
        :return:
        """
        state = self
        for char in key:
            state = state._children.get(char)
            if state is None:
                return None

        return state._value

    def __setitem__(self, key, value):
        """
        魔术方法，定义设置容器中指定元素的行为，相当于 self[key] = value
        :param key:
        :param value:
        :return:
        """

        state = self
        for i, char in enumerate(key):
            if i < len(key) - 1:
                state = state._add_child(char, None, False)
            else:
                state = state._add_child(char, value, True)


if __name__ == '__main__':
    trie = Trie()
    # 增
    trie['自然'] = 'nature'
    trie['自然人'] = 'human'
    trie['自认语言'] = 'language'
    trie['自语'] = 'talk to oneself'
    trie['入门'] = 'introduction'
    trie['出门'] = 'out'

    assert '自然' in trie

    # 删
    trie['自然'] = None
    assert '自然' not in trie

    # 改
    trie['自然语言'] = 'human language'
    assert trie['自然语言'] == 'human language'

    # 查
    assert trie['入门'] == 'introduction'
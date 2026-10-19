"""Prepared words, striding and refusal publication agree with the contract."""

import shutil
import subprocess

import pytest

from merlin.llvmlower.word_group_preparation import emit_u16_group_preparation


def test_compiled_strided_preparation_refreshes_words_and_only_publishes_complete_maximum(tmp_path):
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable")
    source, binary = tmp_path / "group.c", tmp_path / "group"
    group = emit_u16_group_preparation(5, symbol="prepare", conversion="convert", magnitude="key")
    source.write_text('''#include <stdint.h>
      static int convert(float value,uint16_t *word) {
        if(value<0)return -1;
        *word=(uint16_t)value;
        return 0;
      }
      static uint16_t key(uint16_t value){return value & 0x7fffu;}
    ''' + group + '''
      int main(void) {
        const float input[]={1,999,32772,999,3,999,6,999,2,999};
        uint16_t words[6]={0,0,0,0,0,0xa55a}, maximum=123;
        if(prepare(input,2,words,&maximum) || maximum!=6 || words[1]!=32772 || words[5]!=0xa55a)return 1;
        float changed[]={11,1,-1,1,30,1,40,1,50,1};maximum=99;
        if(prepare(changed,2,words,&maximum)!=-1 || maximum!=99 || words[0]!=11 || words[1]!=32772)return 2;
        if(words[5]!=0xa55a)return 3;
        return 0;
      }''')
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("count", [0, -1, True, 1048577])
def test_unknown_or_unbounded_group_extents_refuse(count):
    with pytest.raises(ValueError):
        emit_u16_group_preparation(count, symbol="prepare", conversion="convert", magnitude="key")


@pytest.mark.parametrize("symbol", ["", "x;y", "λ"])
def test_untrusted_c_names_refuse(symbol):
    with pytest.raises(ValueError):
        emit_u16_group_preparation(32, symbol=symbol, conversion="convert", magnitude="key")

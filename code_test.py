from arguments import parser
from data_prepare import DataPrepare

args = parser.parse_args()

prepare = DataPrepare(args, ['1A0G_A_B'])
prepare()
